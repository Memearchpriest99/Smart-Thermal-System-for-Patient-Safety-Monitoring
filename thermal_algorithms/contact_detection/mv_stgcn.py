"""MVSTGCNDetector — Multi-View Spatiotemporal Graph CNN (§ 4.4.3.2).

Three-stage pipeline
--------------------
Stage 1 — Thermal Object Detector + Kalman Tracker
    An embedded ``HumanDetector`` (injected at construction) runs on each
    camera frame.  A Constant-Velocity Kalman filter corrects for the ≈50 ms
    I2C phase shift between Camera-1 and Camera-3, and smooths 1-pixel
    quantisation jitter, producing stable velocity vectors.

Stage 2 — Homographic Projection → Unique Actor Nodes
    Each detection's foot-point is projected to world coordinates through
    H_k.  Cross-camera projections are validated and clustered (ε = 0.5 m)
    to yield unique ActorPosition nodes.

Stage 3 — Spatiotemporal Graph Convolution
    A sliding window of T = 16 frames feeds the ST-GCN:
    - Nodes: one per unique actor per timestep.
    - Spatial edges: fully connected within a timestep (adaptive Gaussian
      adjacency weighted by pairwise distance).
    - Temporal edges: connect the same actor across consecutive timesteps.
    Node feature vector (7-D per node):
        [X_norm, Y_norm, vx, vy, max_temp_norm, mean_temp_norm, bbox_area]
    Output: 2-class probability (Contact / No-Contact) via global avg pool
    and dense classifier.

Persistence gate: alert only if the GCN output > ``conf_threshold`` for at
least ``persistence_frames`` consecutive frames (removes transient noise).

Resolution behavior: 'invariant' — all graph operations use world-space
coordinates from the homographic projection.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import (
    ActorPosition,
    ContactEvent,
    Detection,
    Frame,
    HomographyMatrices,
)
from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewDetections,
    ThreeViewFrames,
)
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections
from thermal_algorithms.contact_detection.multi_view.tracker import PerCameraTracker


# ---------------------------------------------------------------------------
# PyTorch model (lazy import so the rest of the library runs without torch)
# ---------------------------------------------------------------------------

def _build_stgcn_model(node_feat_dim: int, hidden_dim: int, n_layers: int, max_actors: int, T: int):
    """Construct the ST-GCN nn.Module on first use."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _AdaptiveGCNLayer(nn.Module):
        def __init__(self, in_d: int, out_d: int, sigma: float = 1.0) -> None:
            super().__init__()
            self.W = nn.Linear(in_d, out_d, bias=False)
            self.sigma2 = sigma ** 2

        def forward(self, x: "torch.Tensor", dist2: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            # x:     (B, N, in_d)
            # dist2: (B, N, N) squared world-plane distances
            # mask:  (B, N) 1 = present, 0 = padded
            B, N, _ = x.shape
            # Adaptive adjacency A_ij = softmax(-dist2 / σ²) along dim j
            neg_dist = -dist2 / max(self.sigma2, 1e-6)
            neg_dist = neg_dist.masked_fill(mask.unsqueeze(1).expand_as(neg_dist) == 0, -1e9)
            A = F.softmax(neg_dist, dim=2)      # (B, N, N) row-stochastic
            # Add self-loops
            I = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
            A_hat = A + I
            # Symmetric normalisation: D^{-1/2} A_hat D^{-1/2}
            deg = A_hat.sum(dim=2, keepdim=True).clamp(min=1e-6)
            A_norm = A_hat / deg
            # GCN step
            support = torch.bmm(A_norm, x)      # (B, N, in_d)
            out = F.relu(self.W(support))        # (B, N, out_d)
            # Zero-out padded nodes
            out = out * mask.unsqueeze(-1).float()
            return out

    class _ThermalSTGCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dims = [node_feat_dim] + [hidden_dim] * n_layers
            self.gcn_layers = nn.ModuleList([
                _AdaptiveGCNLayer(dims[i], dims[i + 1]) for i in range(n_layers)
            ])
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(
            self,
            features: "torch.Tensor",   # (B, T, N, node_feat_dim)
            positions: "torch.Tensor",  # (B, T, N, 2)
            masks: "torch.Tensor",      # (B, T, N)  bool / float
        ) -> "torch.Tensor":            # (B, 2)
            B, T_len, N, _ = features.shape
            # Compute pairwise squared distances per timestep
            # (B, T, N, 2) → (B*T, N, 2)
            pos_flat = positions.view(B * T_len, N, 2)
            diff = pos_flat.unsqueeze(2) - pos_flat.unsqueeze(1)   # (B*T, N, N, 2)
            dist2 = (diff ** 2).sum(dim=-1)                         # (B*T, N, N)

            feat_flat = features.view(B * T_len, N, -1)
            mask_flat = masks.view(B * T_len, N)

            x = feat_flat
            for layer in self.gcn_layers:
                x = layer(x, dist2, mask_flat)

            # x: (B*T, N, hidden_dim) → (B, T, N, hidden_dim)
            x = x.view(B, T_len, N, -1)

            # Global average pool over valid (actor, timestep) pairs
            valid = masks.unsqueeze(-1).float()  # (B, T, N, 1)
            n_valid = valid.sum(dim=(1, 2)).clamp(min=1)  # (B, 1)
            pooled = (x * valid).sum(dim=(1, 2)) / n_valid  # (B, hidden_dim)

            return self.head(pooled)  # (B, 2)

    return _ThermalSTGCN()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class MVSTGCNDetector(ContactDetector):
    """Multi-View Spatiotemporal GCN contact detector (§ 4.4.3.2)."""

    name = "mv_stgcn_detector"
    is_trainable = True
    resolution_behavior = "invariant"

    def __init__(
        self,
        sensor_profile: Optional[SensorProfile] = None,
        *,
        homography: Optional[HomographyMatrices] = None,
        human_detector=None,             # optional HumanDetector for embedded use
        # Fusion
        epsilon_m: float = 0.5,
        # GCN architecture
        T: int = 16,
        max_actors: int = 5,
        node_feat_dim: int = 7,
        hidden_dim: int = 32,
        n_gcn_layers: int = 2,
        # Persistence gate
        conf_threshold: float = 0.7,
        persistence_frames: int = 3,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 30,
        batch_size: int = 16,
        device: Optional[str] = None,
        random_state: int = 0,
        class_weight: Optional[tuple[float, float]] = None,
    ) -> None:
        """
        class_weight: Optional ``(weight_no_contact, weight_contact)`` passed
            to ``nn.CrossEntropyLoss(weight=...)`` — contact is a small
            minority class in the natural-ratio dataset (see
            data/DATASET_NOTES.md). ``None`` keeps uniform weighting.
        """
        super().__init__(
            sensor_profile=sensor_profile,
            homography=homography,
            epsilon_m=epsilon_m,
            T=T,
            max_actors=max_actors,
            node_feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            n_gcn_layers=n_gcn_layers,
            conf_threshold=conf_threshold,
            persistence_frames=persistence_frames,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            random_state=random_state,
            class_weight=class_weight,
        )
        self._class_weight = class_weight
        self._human_detector = human_detector
        self._epsilon_m = float(epsilon_m)
        self._T = int(T)
        self._max_actors = int(max_actors)
        self._node_feat_dim = int(node_feat_dim)
        self._hidden_dim = int(hidden_dim)
        self._n_gcn_layers = int(n_gcn_layers)
        self._conf_threshold = float(conf_threshold)
        self._persistence_frames = int(persistence_frames)
        self._lr = float(learning_rate)
        self._wd = float(weight_decay)
        self._n_epochs = int(n_epochs)
        self._batch_size = int(batch_size)
        self._device_str: Optional[str] = device
        self._rng = np.random.default_rng(int(random_state))

        # Per-camera Kalman trackers
        self._trackers = [PerCameraTracker() for _ in range(3)]
        self._next_actor_id: int = 0

        # Sliding window: deque of (features_array, positions_array, mask_array)
        # Each entry = one timestep, shape (max_actors, node_feat_dim), etc.
        self._window_features: deque = deque(maxlen=self._T)
        self._window_positions: deque = deque(maxlen=self._T)
        self._window_masks: deque = deque(maxlen=self._T)

        # Persistence counter
        self._persistence_count: int = 0

        # Model (built lazily)
        self._model = None
        self._device = None

    # ---- Model construction -------------------------------------------------

    def _get_model(self):
        if self._model is None:
            import torch
            self._model = _build_stgcn_model(
                self._node_feat_dim, self._hidden_dim, self._n_gcn_layers,
                self._max_actors, self._T,
            )
            dev_str = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = torch.device(dev_str)
            self._model = self._model.to(self._device)
        return self._model, self._device

    # ---- Fit ----------------------------------------------------------------

    def fit(
        self,
        X: Iterable[ThreeViewFrames],
        y: Iterable[ContactEvent] | None = None,
    ) -> "MVSTGCNDetector":
        """Train the ST-GCN on a labelled sequence of frame triplets.

        Args:
            X: Iterable of (Frame0, Frame1, Frame2) triplets in temporal order.
            y: Parallel iterable of ContactEvents (ground-truth labels).
               ``event.any_contact`` is used as the binary label.
        """
        import torch
        import torch.nn as nn

        if y is None:
            raise ValueError("MVSTGCNDetector.fit() requires labelled ContactEvents (y=...).")

        model, device = self._get_model()

        # Materialise and build sliding windows
        examples = list(zip(X, y))
        if not examples:
            raise ValueError("fit() received an empty dataset.")

        windows = self._build_training_windows(examples)
        if not windows:
            raise ValueError(
                f"fit() produced no training windows "
                f"(need at least T={self._T} consecutive frames)."
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr, weight_decay=self._wd)
        weight = (
            torch.tensor(self._class_weight, dtype=torch.float32, device=device)
            if self._class_weight is not None else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight)

        model.train()
        for epoch in range(self._n_epochs):
            order = self._rng.permutation(len(windows))
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(windows), self._batch_size):
                idxs = order[start:start + self._batch_size]
                feat_b, pos_b, mask_b, label_b = self._collate([windows[i] for i in idxs])
                feat_b = feat_b.to(device)
                pos_b = pos_b.to(device)
                mask_b = mask_b.to(device)
                label_b = label_b.to(device)

                logits = model(feat_b, pos_b, mask_b)
                loss = criterion(logits, label_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        self._is_fitted = True
        return self

    # ---- Predict ------------------------------------------------------------

    def predict(
        self,
        X: ThreeViewFrames,
        detections: Optional[ThreeViewDetections] = None,
    ) -> ContactEvent:
        """Run one timestep through the full pipeline.

        If ``detections`` is None and a ``human_detector`` was provided at
        construction, it will be called internally.  Otherwise detections must
        be supplied.
        """
        timestamp = max(f.timestamp for f in X)

        # Stage 1: detections per camera
        if detections is None:
            if self._human_detector is None:
                raise ValueError(
                    "MVSTGCNDetector.predict(): provide pre-computed detections "
                    "or inject a human_detector at construction."
                )
            detections = tuple(self._human_detector.predict(f) for f in X)

        # Stage 1b: Kalman tracking (forward-predict from cam0 to cam2 timing)
        track_info = []
        for cam_id, (dets, tracker) in enumerate(zip(detections, self._trackers)):
            centroids = [d.foot_point for d in dets]
            tracker.predict_all()
            live = tracker.update(centroids)
            track_info.append((cam_id, dets, live))

        # Stage 2: homographic fusion → actor positions
        if self._homography is not None:
            actors, self._next_actor_id = fuse_detections(
                list(detections), self._homography,
                epsilon_m=self._epsilon_m,
                next_track_id=self._next_actor_id,
            )
        else:
            actors = []

        # Build node features for this timestep
        feat, pos, mask = self._actors_to_node_features(actors, detections)
        self._window_features.append(feat)
        self._window_positions.append(pos)
        self._window_masks.append(mask)

        # Stage 3: GCN inference (only when window is full)
        if not self._is_fitted or len(self._window_features) < self._T:
            return ContactEvent(
                actors=tuple(actors),
                pairs_in_contact=(),
                timestamp=timestamp,
                confidence=0.0,
                debug={"status": "buffer_filling", "n_buffered": len(self._window_features)},
            )

        confidence = self._run_gcn_inference()
        in_contact = confidence > self._conf_threshold

        if in_contact:
            self._persistence_count += 1
        else:
            self._persistence_count = 0

        alerted = in_contact and self._persistence_count >= self._persistence_frames

        pairs: tuple[tuple[int, int], ...] = ()
        if alerted and len(actors) >= 2:
            import math
            pairs = tuple(
                (i, j)
                for i in range(len(actors))
                for j in range(i + 1, len(actors))
                if math.sqrt(sum((a - b) ** 2 for a, b in zip(actors[i].world_xy, actors[j].world_xy)))
                   < self._epsilon_m * 2
            )

        return ContactEvent(
            actors=tuple(actors),
            pairs_in_contact=pairs,
            timestamp=timestamp,
            confidence=float(confidence),
        )

    def reset(self) -> None:
        for t in self._trackers:
            t.reset()
        self._window_features.clear()
        self._window_positions.clear()
        self._window_masks.clear()
        self._persistence_count = 0
        self._next_actor_id = 0

    # ---- Internal -----------------------------------------------------------

    def _actors_to_node_features(
        self,
        actors: list[ActorPosition],
        detections: ThreeViewDetections,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build padded feature/position/mask arrays for one timestep.

        Returns arrays of shape (max_actors, node_feat_dim), (max_actors, 2),
        (max_actors,).
        """
        N = self._max_actors
        feat = np.zeros((N, self._node_feat_dim), dtype=np.float32)
        pos = np.zeros((N, 2), dtype=np.float32)
        mask = np.zeros(N, dtype=np.float32)

        # Gather per-actor thermal features from detections
        all_dets: list[Detection] = [d for dets in detections for d in dets]
        max_temp_all = max((d.thermal_features.get("max_temp", 0) for d in all_dets
                           if d.thermal_features), default=37.0)
        mean_temp_all = np.mean([d.thermal_features.get("mean_temp", 37.0)
                                 for d in all_dets if d.thermal_features] or [37.0])

        for idx, actor in enumerate(actors[:N]):
            x, y = actor.world_xy
            # Normalise world position to roughly [-1, 1] (room ≈ 5 m)
            pos[idx] = [x / 5.0, y / 5.0]
            mask[idx] = 1.0
            # Velocities from tracker (if available via source cameras)
            vx, vy = 0.0, 0.0
            for cam_id in actor.source_camera_ids:
                tracker = self._trackers[cam_id]
                for track in tracker.tracks.values():
                    vx, vy = track.velocity
                    break
            # Thermal features: use best-matching detection
            max_t = max_temp_all / 100.0 if max_temp_all else 0.37
            mean_t = float(mean_temp_all) / 100.0
            area = float(all_dets[idx % max(len(all_dets), 1)].area) / 1000.0 \
                if all_dets else 0.0
            feat[idx] = [pos[idx, 0], pos[idx, 1], vx / 10.0, vy / 10.0,
                         max_t, mean_t, area]

        return feat, pos, mask

    def _run_gcn_inference(self) -> float:
        import torch
        import torch.nn.functional as F

        model, device = self._get_model()
        model.eval()

        feat_arr = np.stack(list(self._window_features), axis=0)   # (T, N, D)
        pos_arr = np.stack(list(self._window_positions), axis=0)   # (T, N, 2)
        mask_arr = np.stack(list(self._window_masks), axis=0)      # (T, N)

        feat_t = torch.from_numpy(feat_arr).unsqueeze(0).to(device)   # (1, T, N, D)
        pos_t = torch.from_numpy(pos_arr).unsqueeze(0).to(device)     # (1, T, N, 2)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).to(device)   # (1, T, N)

        with torch.no_grad():
            logits = model(feat_t, pos_t, mask_t)                     # (1, 2)
            prob = F.softmax(logits, dim=1)[0, 1].item()
        return float(prob)

    def _build_training_windows(self, examples):
        """Extract sliding windows of length T from the labelled sequence."""
        windows = []
        for start in range(len(examples) - self._T + 1):
            window = examples[start:start + self._T]
            frames_seq = [w[0] for w in window]
            contact_seq = [w[1] for w in window]
            label = 1 if contact_seq[-1].any_contact else 0
            # Build per-timestep node features
            feats, poses, masks = [], [], []
            for triplet, contact in zip(frames_seq, contact_seq):
                dets_dummy = ([], [], [])  # no embedded detector during training
                actors = list(contact.actors)
                f, p, m = self._actors_to_node_features(actors, dets_dummy)
                feats.append(f)
                poses.append(p)
                masks.append(m)
            windows.append((
                np.stack(feats, axis=0),   # (T, N, D)
                np.stack(poses, axis=0),   # (T, N, 2)
                np.stack(masks, axis=0),   # (T, N)
                label,
            ))
        return windows

    def _collate(self, batch):
        import torch
        feat_b = torch.from_numpy(np.stack([w[0] for w in batch], axis=0))  # (B, T, N, D)
        pos_b = torch.from_numpy(np.stack([w[1] for w in batch], axis=0))   # (B, T, N, 2)
        mask_b = torch.from_numpy(np.stack([w[2] for w in batch], axis=0))  # (B, T, N)
        label_b = torch.tensor([w[3] for w in batch], dtype=torch.long)     # (B,)
        return feat_b, pos_b, mask_b, label_b

    # ---- Persistence --------------------------------------------------------

    def _state_dict(self) -> dict:
        if self._model is None:
            return {}
        # Homography lives on ContactDetector (outside _params), so it must be
        # persisted here explicitly — without it the fusion front-end reloads
        # dead (zero actors → constant confidence).
        return {"model_state": self._model.state_dict(),
                "homography": self._homography}

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        import torch
        model, device = self._get_model()
        sd = {k: v.to(device) if hasattr(v, "to") else v
              for k, v in state["model_state"].items()}
        model.load_state_dict(sd)
        if state.get("homography") is not None:
            self._homography = state["homography"]
        self._is_fitted = True
