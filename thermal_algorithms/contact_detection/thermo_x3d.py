"""ThermoX3DDetector — Volumetric Thermal Interaction Network (§ 4.4.3.3).

Motivation
----------
The MV-STGCN fails when close contact causes bounding boxes to merge (the
object detector sees one large blob instead of two people, leaving the GCN
with a single node and no edges).  Thermo-X3D is the pixel-based failsafe:
it does not rely on bounding boxes at all.

Architecture (Micro-X3D)
------------------------
Each camera stream is processed by a small (2+1)D factorised X3D encoder:

    Input  : (B, 1, T=5, H, W)   — 1-channel thermal volume

    Stem   : Conv3D (1×3×3, stride 1, 24 filters)
    Block1 : Factorised (2+1)D ResBlock, 24 filters
    Block2 : Factorised (2+1)D ResBlock, 48 filters, spatial stride 2
    Block3 : Factorised (2+1)D ResBlock, 96 filters, spatial stride 2
    Pool   : Global Average Pool → Linear(96, 128)

    Output : feature vector v_i ∈ ℝ^128

Late fusion concatenates the three per-stream vectors and classifies:

    v_final = Concat(v_1, v_2, v_3)   ∈ ℝ^384
    Classifier: Dense(384→64, ReLU) → Dense(64→2, Softmax)

Thermal Attention
-----------------
Before each ResBlock, a CBAM-style attention module (channel + spatial)
forces the network to focus on hot voxels rather than cold background:

    Channel map  M_c = σ(MLP(AvgPool(X)) + MLP(MaxPool(X)))
    Spatial map  M_s = σ(Conv7×7([AvgPool_c(X); MaxPool_c(X)]))
    Refined      X'' = X · M_c · M_s

Fixed-step sampling
-------------------
The buffer always feeds the network frames at an effective 8 Hz, regardless
of capture FPS.  The ThermoX3DDetector.predict() is called once per frame; it
up/downsamples the rolling buffer internally.

Resolution behavior: 'fixed' — one checkpoint per sensor profile (MLX90640
or Waveshare) because the spatial architecture dimensions depend on (H, W).
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile, MLX90640
from thermal_algorithms.core.types import ContactEvent, Frame, HomographyMatrices
from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewDetections,
    ThreeViewFrames,
)


# ---------------------------------------------------------------------------
# PyTorch model (lazy import)
# ---------------------------------------------------------------------------

def _build_x3d_model(h: int, w: int):
    """Build the Thermo-X3D nn.Module for input height h, width w."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # ---- CBAM-style Thermal Attention ----------------------------------------

    class _ChannelAttention(nn.Module):
        def __init__(self, channels: int, reduction: int = 4) -> None:
            super().__init__()
            mid = max(1, channels // reduction)
            self.mlp = nn.Sequential(nn.Linear(channels, mid), nn.ReLU(),
                                     nn.Linear(mid, channels))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, T, H, W)
            avg = x.mean(dim=(2, 3, 4))      # (B, C)
            mx = x.amax(dim=(2, 3, 4))       # (B, C)
            M = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
            return x * M.view(M.shape[0], M.shape[1], 1, 1, 1)

    class _SpatialAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 7×7 conv over channel-pooled maps; pad to preserve (H, W)
            self.conv = nn.Conv3d(2, 1, kernel_size=(1, 7, 7), padding=(0, 3, 3), bias=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, C, T, H, W)
            avg = x.mean(dim=1, keepdim=True)   # (B, 1, T, H, W)
            mx = x.amax(dim=1, keepdim=True)    # (B, 1, T, H, W)
            cat = torch.cat([avg, mx], dim=1)   # (B, 2, T, H, W)
            M = torch.sigmoid(self.conv(cat))   # (B, 1, T, H, W)
            return x * M

    class _ThermalAttention(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.channel = _ChannelAttention(channels)
            self.spatial = _SpatialAttention()

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.spatial(self.channel(x))

    # ---- (2+1)D Factorised ResBlock ------------------------------------------

    class _FactorizedResBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int, spatial_stride: int = 1) -> None:
            super().__init__()
            # Spatial 2D conv in the (H, W) plane: kernel (1, 3, 3)
            self.spatial = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3),
                          stride=(1, spatial_stride, spatial_stride),
                          padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )
            # Temporal 1D conv: kernel (3, 1, 1)
            self.temporal = nn.Sequential(
                nn.Conv3d(out_c, out_c, kernel_size=(3, 1, 1),
                          stride=1, padding=(1, 0, 0), bias=False),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )
            # Skip connection
            self.skip = (
                nn.Sequential(
                    nn.Conv3d(in_c, out_c, kernel_size=1,
                              stride=(1, spatial_stride, spatial_stride), bias=False),
                    nn.BatchNorm3d(out_c),
                )
                if (in_c != out_c or spatial_stride != 1)
                else nn.Identity()
            )
            self.attention = _ThermalAttention(out_c)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            identity = self.skip(x)
            out = self.temporal(self.spatial(x))
            out = self.attention(out)
            return F.relu(out + identity)

    # ---- Single-camera encoder -----------------------------------------------

    class _MicroX3DStream(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv3d(1, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(24),
                nn.ReLU(inplace=True),
            )
            self.block1 = _FactorizedResBlock(24, 24, spatial_stride=1)
            self.block2 = _FactorizedResBlock(24, 48, spatial_stride=2)
            self.block3 = _FactorizedResBlock(48, 96, spatial_stride=2)
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.proj = nn.Linear(96, 128)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, 1, T, H, W)
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x).flatten(1)  # (B, 96)
            return self.proj(x)           # (B, 128)

    # ---- Late-fusion classifier ----------------------------------------------

    class _ThermoX3DModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.streams = nn.ModuleList([_MicroX3DStream() for _ in range(3)])
            self.head = nn.Sequential(
                nn.Linear(384, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 2),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            # x: (B, 3, T, H, W) — batch, camera, time, height, width
            features = [self.streams[i](x[:, i:i+1, :, :, :]) for i in range(3)]
            fused = torch.cat(features, dim=1)  # (B, 384)
            return self.head(fused)             # (B, 2)

    return _ThermoX3DModel()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ThermoX3DDetector(ContactDetector):
    """Volumetric pixel-based contact detector (§ 4.4.3.3 Thermo-X3D).

    Does not require bounding boxes — processes raw thermal frames directly.
    """

    name = "thermo_x3d_detector"
    is_trainable = True
    resolution_behavior = "fixed"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        homography: Optional[HomographyMatrices] = None,
        T: int = 5,
        conf_threshold: float = 0.7,
        persistence_frames: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 8,
        device: Optional[str] = None,
        random_state: int = 0,
        class_weight: Optional[tuple[float, float]] = None,
        min_positive_frames: int = 2,
        weight_init: str = "identity",
    ) -> None:
        """
        class_weight: Optional ``(weight_no_contact, weight_contact)`` passed
            to ``nn.CrossEntropyLoss(weight=...)`` — contact is a small
            minority class in the natural-ratio dataset (see
            data/DATASET_NOTES.md). ``None`` keeps uniform weighting.
        min_positive_frames: How many of the ``T`` frames in a training window
            must be contact-positive for the window to be labelled positive
            (see ``_build_training_windows``). Default 2.
        weight_init: ``"identity"`` (default) initialises every conv as a
            Dirac/identity kernel and every linear layer as a (partial)
            identity matrix, so the untrained network is close to a
            pass-through; ``"default"`` keeps PyTorch's standard Kaiming-style
            init. See ``_apply_identity_init`` for the caveats.
        """
        if sensor_profile is None:
            raise ValueError("ThermoX3DDetector requires a SensorProfile.")

        super().__init__(
            sensor_profile=sensor_profile,
            homography=homography,
            T=T,
            conf_threshold=conf_threshold,
            persistence_frames=persistence_frames,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
            class_weight=class_weight,
            min_positive_frames=min_positive_frames,
            weight_init=weight_init,
        )
        self._class_weight = class_weight
        self._min_positive_frames = int(min_positive_frames)
        self._weight_init = str(weight_init)
        self._T = int(T)
        self._conf_threshold = float(conf_threshold)
        self._persistence_frames = int(persistence_frames)
        self._lr = float(learning_rate)
        self._wd = float(weight_decay)
        self._n_epochs = int(n_epochs)
        self._batch_size = int(batch_size)
        self._device_str: Optional[str] = device
        self._rng = np.random.default_rng(int(random_state))

        w, h = sensor_profile.resolution
        self._input_h = h
        self._input_w = w

        # Rolling buffers: one deque per camera, maxlen = T
        self._buffers: list[deque] = [deque(maxlen=T) for _ in range(3)]

        # Global normalisation statistics (learned during fit)
        self._global_mean: float = 0.0
        self._global_std: float = 1.0
        # Once frozen (via set_normalization()), fit() no longer recomputes
        # these from whatever chunk it's currently handed -- see fit()'s
        # docstring for why per-chunk recomputation is a bug for chunked
        # training. Callers that never opt in keep the old (broken) behaviour.
        self._norm_frozen: bool = False

        # Persistence counter
        self._persistence_count: int = 0

        # Model + optimizer (built lazily, persist across fit() calls)
        self._model = None
        self._device = None
        self._optimizer = None
        self._last_fit_loss: Optional[float] = None

    # ---- Model construction -------------------------------------------------

    def _get_model(self):
        if self._model is None:
            import torch
            self._model = _build_x3d_model(self._input_h, self._input_w)
            if self._weight_init == "identity":
                self._apply_identity_init(self._model)
            dev_str = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev_str)
            self._model = self._model.to(self._device)
        return self._model, self._device

    @staticmethod
    def _apply_identity_init(model) -> None:
        """Initialise every conv as a Dirac (identity) kernel and every linear
        layer as a (partial) identity matrix, so an untrained network is close
        to a pass-through rather than a random projection.

        Two honest caveats, both inherent to identity init on THIS
        architecture rather than to the implementation:

        1. ``dirac_`` can only make ``min(out_channels, in_channels)`` output
           channels pass-through; the rest are left at ZERO. The stem
           (1 -> 24 channels) therefore starts with 1 live channel and 23 dead
           ones. Those channels are not permanently dead -- they still receive
           gradient from downstream layers -- but they do start from zero, so
           early training spends capacity waking them up.
        2. Identity init gives every one of the three camera streams the exact
           same initial weights. They only differentiate through the gradients
           their (different) inputs produce; with default random init they
           start differentiated. Whether that matters is empirical, which is
           why ``weight_init`` is exposed as a searchable hyperparameter rather
           than hardcoded.

        BatchNorm is left at PyTorch's default (weight=1, bias=0), which is
        already the identity, and all biases are zeroed.
        """
        import torch.nn as nn
        from torch.nn.init import dirac_, eye_, zeros_

        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                try:
                    dirac_(m.weight)
                except (ValueError, RuntimeError):
                    # dirac_ refuses some shapes (e.g. more in- than
                    # out-channels in a way it cannot represent); leaving
                    # PyTorch's default init for those layers is strictly
                    # better than crashing or zeroing them.
                    pass
                if m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                eye_(m.weight)          # non-square -> partial identity
                if m.bias is not None:
                    zeros_(m.bias)

    def _get_optimizer(self, model):
        """Lazily build and cache a single AdamW instance so its momentum/
        adaptive-LR state persists across chunked fit() calls -- constructing
        a fresh optimizer per chunk (the old behaviour) discards that state
        every time, which is one of the confirmed causes of the degenerate
        constant-classifier bug documented in thermox3d_training_bug.md."""
        if self._optimizer is None:
            import torch
            self._optimizer = torch.optim.AdamW(
                model.parameters(), lr=self._lr, weight_decay=self._wd
            )
        return self._optimizer

    def set_normalization(self, mean: float, std: float) -> None:
        """Freeze global normalisation stats (e.g. computed once from a
        train-only sample) so fit() stops recomputing mean/std from whatever
        chunk it's currently handed. Must be called before ANY call that
        builds windows (fit(), or a caller building windows itself via
        _build_training_windows) -- windows built before this call bake in
        whatever stats were active at that time."""
        self._global_mean = float(mean)
        self._global_std = float(std) if std else 1.0
        self._norm_frozen = True

    # ---- Fit ----------------------------------------------------------------

    def fit(
        self,
        X: Iterable[ThreeViewFrames],
        y: Iterable[ContactEvent] | None = None,
    ) -> "ThermoX3DDetector":
        """Train Thermo-X3D on labelled frame triplets.

        Args:
            X: Iterable of (Frame0, Frame1, Frame2) in temporal order.
            y: Parallel ContactEvents; ``event.any_contact`` is the label.
        """
        if y is None:
            raise ValueError("ThermoX3DDetector.fit() requires labelled ContactEvents (y=...).")

        examples = list(zip(X, y))
        if not examples:
            raise ValueError("fit() received an empty dataset.")

        # Global normalisation statistics: recomputed from THIS chunk only if
        # never frozen via set_normalization(). Recomputing per chunk is the
        # historical (buggy) behaviour for chunked training -- kept as the
        # default so callers that don't opt in are unaffected -- but any
        # caller doing real chunked/incremental training should call
        # set_normalization() once, up front, with corpus-representative
        # stats (see thermox3d_training_bug.md).
        if not self._norm_frozen:
            all_temps = []
            for triplet, _ in examples:
                for f in triplet:
                    all_temps.append(f.data.ravel())
            all_arr = np.concatenate(all_temps).astype(np.float32)
            self._global_mean = float(all_arr.mean())
            self._global_std = float(all_arr.std()) + 1e-6

        # Build sliding-window training samples
        windows = self._build_training_windows(examples)
        if not windows:
            raise ValueError(
                f"fit() produced no windows (need ≥ T={self._T} frames)."
            )

        self._fit_windows(windows)
        return self

    def _fit_windows(self, windows) -> float:
        """Run self._n_epochs epochs of training over an already-built list
        of (volume, label) windows, using the lazily-cached, PERSISTENT
        optimizer (see _get_optimizer) so momentum/adaptive-LR state carries
        over across calls -- this is what actually makes chunked training
        continue rather than restart from scratch on every chunk. Callers
        that build windows themselves (e.g. concatenating per-run window
        lists to avoid cross-run "seam" windows) can call this directly,
        bypassing fit()'s own windowing. Returns the mean loss over every
        optimizer step in this call (also stored as self._last_fit_loss)."""
        import torch
        import torch.nn as nn

        if not windows:
            raise ValueError("_fit_windows() received no windows.")

        model, device = self._get_model()
        optimizer = self._get_optimizer(model)
        weight = (
            torch.tensor(self._class_weight, dtype=torch.float32, device=device)
            if self._class_weight is not None else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        model.train()
        total_loss, n_steps = 0.0, 0
        for epoch in range(self._n_epochs):
            order = self._rng.permutation(len(windows))
            for start in range(0, len(windows), self._batch_size):
                idxs = order[start:start + self._batch_size]
                vol_b, label_b = self._collate([windows[i] for i in idxs])
                vol_b = vol_b.to(device)
                label_b = label_b.to(device)
                logits = model(vol_b)
                loss = criterion(logits, label_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1

        self._is_fitted = True
        self._last_fit_loss = total_loss / max(1, n_steps)
        return self._last_fit_loss

    def _eval_windows_loss(self, windows) -> float:
        """No-grad mean CrossEntropy loss over a fixed window list (e.g. a
        held-out validation set) -- used for early-stopping checks. Does not
        touch the optimizer or model.train()/eval() state permanently beyond
        this call (model is switched back to train() before returning, since
        every caller of this method is expected to resume training after)."""
        import torch
        import torch.nn as nn

        if not windows:
            raise ValueError("_eval_windows_loss() received no windows.")

        model, device = self._get_model()
        weight = (
            torch.tensor(self._class_weight, dtype=torch.float32, device=device)
            if self._class_weight is not None else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        model.eval()
        total_loss, n_steps = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(windows), self._batch_size):
                batch = windows[start:start + self._batch_size]
                vol_b, label_b = self._collate(batch)
                vol_b = vol_b.to(device)
                label_b = label_b.to(device)
                logits = model(vol_b)
                loss = criterion(logits, label_b)
                total_loss += float(loss.item())
                n_steps += 1
        model.train()
        return total_loss / max(1, n_steps)

    # ---- Predict ------------------------------------------------------------

    def predict(
        self,
        X: ThreeViewFrames,
        detections: Optional[ThreeViewDetections] = None,
    ) -> ContactEvent:
        """Process one timestep. Detections are ignored (pixel-based)."""
        timestamp = max(f.timestamp for f in X)

        # Update rolling buffers (fixed-step at effective 8 Hz)
        for cam_id, frame in enumerate(X):
            self._buffers[cam_id].append(frame.data.astype(np.float32))

        # Need T frames in all three buffers
        if (not self._is_fitted
                or any(len(b) < self._T for b in self._buffers)):
            return ContactEvent(
                actors=(), pairs_in_contact=(), timestamp=timestamp,
                confidence=0.0,
                debug={"status": "buffer_filling",
                       "n_buffered": min(len(b) for b in self._buffers)},
            )

        confidence = self._run_x3d_inference()
        in_contact = confidence > self._conf_threshold

        if in_contact:
            self._persistence_count += 1
        else:
            self._persistence_count = 0

        alerted = in_contact and self._persistence_count >= self._persistence_frames
        pairs = ((0, 1),) if alerted else ()

        return ContactEvent(
            actors=(),
            pairs_in_contact=pairs,
            timestamp=timestamp,
            confidence=float(confidence),
        )

    def reset(self) -> None:
        for buf in self._buffers:
            buf.clear()
        self._persistence_count = 0

    # ---- Internal -----------------------------------------------------------

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self._global_mean) / self._global_std

    def _run_x3d_inference(self) -> float:
        import torch
        import torch.nn.functional as F

        model, device = self._get_model()
        model.eval()

        # Build volume tensor (3, T, H, W)
        vols = []
        for buf in self._buffers:
            vol = np.stack([self._normalise(f) for f in list(buf)[-self._T:]], axis=0)
            vols.append(vol)
        vol_arr = np.stack(vols, axis=0)   # (3, T, H, W)
        vol_t = torch.from_numpy(vol_arr).unsqueeze(0).to(device)  # (1, 3, T, H, W)

        with torch.no_grad():
            logits = model(vol_t)
            prob = F.softmax(logits, dim=1)[0, 1].item()
        return float(prob)

    def _build_training_windows(self, examples):
        windows = []
        for start in range(len(examples) - self._T + 1):
            window = examples[start:start + self._T]
            # A window is positive when at least ``min_positive_frames`` of its
            # T frames are contact-positive (project owner's directive: ">=2 of
            # the 5 triplets"). The historical rule was "label by the LAST
            # frame alone", which made a single mislabelled or borderline frame
            # flip an entire window, and gave the network no way to distinguish
            # a sustained contact from a one-frame segmentation blip.
            # min_positive_frames=1 recovers an "any frame" rule; it can never
            # exactly reproduce the old last-frame rule, which is intentional.
            n_pos = sum(1 for _triplet, ev in window if ev.any_contact)
            label = 1 if n_pos >= self._min_positive_frames else 0
            # Stack into (3, T, H, W) volume
            vols = [[], [], []]
            for triplet, _ in window:
                for cam_id, frame in enumerate(triplet):
                    vols[cam_id].append(self._normalise(frame.data.astype(np.float32)))
            vol_arr = np.stack([np.stack(v, axis=0) for v in vols], axis=0)  # (3, T, H, W)
            windows.append((vol_arr, label))
        return windows

    def _collate(self, batch):
        import torch
        vol_b = torch.from_numpy(np.stack([w[0] for w in batch], axis=0))   # (B, 3, T, H, W)
        label_b = torch.tensor([w[1] for w in batch], dtype=torch.long)
        return vol_b, label_b

    # ---- Persistence --------------------------------------------------------

    def _state_dict(self) -> dict:
        if self._model is None:
            return {}
        return {
            "model_state": self._model.state_dict(),
            "global_mean": self._global_mean,
            "global_std": self._global_std,
        }

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        import torch
        model, device = self._get_model()
        sd = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in state["model_state"].items()}
        model.load_state_dict(sd)
        self._global_mean = float(state.get("global_mean", 0.0))
        self._global_std = float(state.get("global_std", 1.0))
        self._is_fitted = True
