"""MobileNetSSDDetector - public deep-learning human detector (Section 4.4.2.3).

Wraps MicroMobileNetSSD with a `HumanDetector` interface: fit/predict/save/load.

PyTorch is imported lazily so the rest of the library remains importable on
machines where torch isn't installed. Only `fit()` and `predict()` actually
require torch; construction, save/load of un-fitted instances, and class
metadata work without it.

The user's recommendation for the project: two separate checkpoints per
architecture (one for each sensor profile). This class is declared
`resolution_behavior='fixed'` and the CheckpointRegistry maps it accordingly.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.human_detection.base import HumanDetector


# Per-profile architecture selection lives in mobilenet_ssd_model. We do not
# import torch at module top-level so the rest of the library remains usable
# on machines without it.

def _pick_configs(profile: SensorProfile):
    """Return (BackboneConfig, SSDConfig) for the given sensor profile."""
    from thermal_algorithms.human_detection.mobilenet_ssd_model import (
        MLX90640_BACKBONE, MLX90640_SSD,
        WAVESHARE_BACKBONE, WAVESHARE_SSD,
    )
    name = profile.name
    if name == "MLX90640":
        return MLX90640_BACKBONE, MLX90640_SSD
    if name == "Waveshare_26984":
        return WAVESHARE_BACKBONE, WAVESHARE_SSD
    raise ValueError(
        f"No MobileNet-SSD config registered for sensor profile {name!r}. "
        f"Add it to thermal_algorithms.human_detection.mobilenet_ssd_model."
    )


class MobileNetSSDDetector(HumanDetector):
    """MicroMobileNet-SSD human detector with per-profile checkpoints."""

    name = "mobilenet_ssd_detector"
    is_trainable = True
    resolution_behavior = "fixed"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        # Inference
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 50,
        batch_size: int = 16,
        iou_pos_threshold: float = 0.5,
        iou_neg_threshold: float = 0.4,
        neg_pos_ratio: float = 3.0,
        loc_loss_weight: float = 1.0,
        # Augmentation
        augment_hflip: bool = True,
        augment_shift_max: float = 0.10,
        # System
        device: Optional[str] = None,
        random_state: int = 0,
    ) -> None:
        if sensor_profile is None:
            raise ValueError("MobileNetSSDDetector requires a SensorProfile.")

        super().__init__(
            sensor_profile=sensor_profile,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            iou_pos_threshold=iou_pos_threshold,
            iou_neg_threshold=iou_neg_threshold,
            neg_pos_ratio=neg_pos_ratio,
            loc_loss_weight=loc_loss_weight,
            augment_hflip=augment_hflip,
            augment_shift_max=augment_shift_max,
            device=device,
            random_state=random_state,
        )

        self._score_threshold = float(score_threshold)
        self._nms_iou = float(nms_iou_threshold)
        self._lr = float(learning_rate)
        self._weight_decay = float(weight_decay)
        self._n_epochs = int(n_epochs)
        self._batch_size = int(batch_size)
        self._iou_pos = float(iou_pos_threshold)
        self._iou_neg = float(iou_neg_threshold)
        self._neg_pos_ratio = float(neg_pos_ratio)
        self._loc_weight = float(loc_loss_weight)
        self._augment_hflip = bool(augment_hflip)
        self._augment_shift_max = float(augment_shift_max)
        self._device_str: Optional[str] = device
        self._rng = np.random.default_rng(int(random_state))

        # Lazily-constructed nn.Module (only created when fit/predict is called).
        self._model = None

    # ---- Lazy model construction --------------------------------------

    def _build_model(self, device=None):
        """Construct the MicroMobileNetSSD nn.Module on the chosen device."""
        import torch
        from thermal_algorithms.human_detection.mobilenet_ssd_model import (
            MicroMobileNetSSD,
        )

        backbone_cfg, ssd_cfg = _pick_configs(self.sensor_profile)
        w, h = self.sensor_profile.resolution     # SensorProfile.resolution is (W, H)
        input_size = (h, w)
        model = MicroMobileNetSSD(
            input_size=input_size,
            backbone_config=backbone_cfg,
            ssd_config=ssd_cfg,
        )

        if device is None:
            device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        model = model.to(device)
        model.build_anchors(device=device)
        return model, device

    @property
    def model(self):
        """The underlying nn.Module, building it on first access if needed."""
        if self._model is None:
            self._model, self._device = self._build_model()
        return self._model

    # ---- Augmentation -------------------------------------------------

    def _augment(self, frame_data: np.ndarray, dets: list[Detection]) -> tuple[np.ndarray, list[Detection]]:
        h, w = frame_data.shape
        data = frame_data.copy()
        new_dets = list(dets)

        # Random horizontal flip
        if self._augment_hflip and self._rng.random() < 0.5:
            data = np.ascontiguousarray(data[:, ::-1])
            new_dets = [
                Detection(
                    bbox=(w - (d.bbox[0] + d.bbox[2]), d.bbox[1], d.bbox[2], d.bbox[3]),
                    score=d.score, class_id=d.class_id, camera_id=d.camera_id,
                )
                for d in new_dets
            ]

        # Random small shift (translate + zero-pad)
        if self._augment_shift_max > 0:
            dx = int(self._rng.integers(-int(w * self._augment_shift_max),
                                        int(w * self._augment_shift_max) + 1))
            dy = int(self._rng.integers(-int(h * self._augment_shift_max),
                                        int(h * self._augment_shift_max) + 1))
            if dx != 0 or dy != 0:
                pad_val = float(data.mean())
                shifted = np.full_like(data, pad_val)
                src_x0 = max(0, -dx); src_y0 = max(0, -dy)
                src_x1 = min(w, w - dx); src_y1 = min(h, h - dy)
                dst_x0 = max(0, dx); dst_y0 = max(0, dy)
                dst_x1 = dst_x0 + (src_x1 - src_x0)
                dst_y1 = dst_y0 + (src_y1 - src_y0)
                shifted[dst_y0:dst_y1, dst_x0:dst_x1] = data[src_y0:src_y1, src_x0:src_x1]
                data = shifted
                new_dets = [
                    Detection(
                        bbox=(d.bbox[0] + dx, d.bbox[1] + dy, d.bbox[2], d.bbox[3]),
                        score=d.score, class_id=d.class_id, camera_id=d.camera_id,
                    )
                    for d in new_dets
                ]

        # Per-frame z-score normalization (robust to ambient temperature drift).
        mu = float(data.mean())
        sigma = float(data.std()) + 1e-6
        data = (data - mu) / sigma
        return data, new_dets

    def _normalize(self, frame_data: np.ndarray) -> np.ndarray:
        """Z-score normalization used at inference (no augmentation)."""
        mu = float(frame_data.mean())
        sigma = float(frame_data.std()) + 1e-6
        return (frame_data - mu) / sigma

    # ---- Helpers to build training tensors -----------------------------

    def _detections_to_cxcywh(self, dets: list[Detection]):
        """Convert a Detection list to a (G, 4) tensor in cxcywh.

        Only class_id == 1 (person) is included; class_id == 0 (fire from the
        YOLO labels) is ignored - this detector is human-only.
        """
        import torch
        boxes = []
        for d in dets:
            if d.class_id != 1:
                continue
            x, y, w, h = d.bbox
            if w <= 0 or h <= 0:
                continue
            boxes.append([x + w / 2.0, y + h / 2.0, w, h])
        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    # ---- Public API ----------------------------------------------------

    def fit(
        self,
        X,                                       # iterable of (Frame, list[Detection])
        y=None,
        *,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> "MobileNetSSDDetector":
        """Train the model on a labelled dataset.

        Args
        ----
        X         : iterable yielding (Frame, list[Detection]) tuples.
                    Typically a FrameLevelDataset, but any iterable works.
        n_epochs  : Override the configured number of epochs.
        verbose   : Print per-epoch loss.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset
        from thermal_algorithms.human_detection.mobilenet_ssd_anchors import ssd_loss

        # Build/refresh the model on the desired device.
        if self._model is None:
            self._model, self._device = self._build_model()
        model = self._model
        device = self._device

        # Materialize the input list so we can iterate it multiple times.
        examples = list(X)
        if not examples:
            raise ValueError("fit() received an empty dataset.")

        def collate(batch_indices):
            imgs = []
            targets = []
            for i in batch_indices:
                frame, dets = examples[i]
                self._validate_shape(frame.data)
                data, aug_dets = self._augment(frame.data, dets)
                imgs.append(torch.from_numpy(data.astype(np.float32)))
                targets.append(self._detections_to_cxcywh(aug_dets))
            imgs = torch.stack(imgs, dim=0).unsqueeze(1)        # (B, 1, H, W)
            return imgs, targets

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self._lr, weight_decay=self._weight_decay,
        )
        anchors = model.anchors

        n_epochs = n_epochs if n_epochs is not None else self._n_epochs
        model.train()
        n = len(examples)
        for epoch in range(n_epochs):
            order = self._rng.permutation(n)
            total_loss = 0.0
            total_cls = 0.0
            total_loc = 0.0
            n_batches = 0
            for start in range(0, n, self._batch_size):
                idxs = order[start:start + self._batch_size]
                imgs, targets = collate(list(idxs))
                imgs = imgs.to(device)
                losses = ssd_loss(
                    *model(imgs), anchors,
                    targets=targets,
                    iou_pos_threshold=self._iou_pos,
                    iou_neg_threshold=self._iou_neg,
                    neg_pos_ratio=self._neg_pos_ratio,
                    loc_loss_weight=self._loc_weight,
                )
                optimizer.zero_grad()
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += float(losses["loss"].item())
                total_cls += float(losses["cls_loss"].item())
                total_loc += float(losses["loc_loss"].item())
                n_batches += 1

            if verbose:
                print(
                    f"epoch {epoch + 1:>3}/{n_epochs}  "
                    f"loss={total_loss / max(1, n_batches):.4f}  "
                    f"cls={total_cls / max(1, n_batches):.4f}  "
                    f"loc={total_loc / max(1, n_batches):.4f}"
                )

        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> list[Detection]:
        if not self._is_fitted:
            raise RuntimeError("MobileNetSSDDetector.predict() called before fit().")
        self._validate_shape(X.data)

        import torch
        from thermal_algorithms.human_detection.mobilenet_ssd_anchors import (
            decode_predictions, nms_xyxy,
        )

        model = self._model
        device = self._device

        model.eval()
        with torch.no_grad():
            data = self._normalize(X.data).astype(np.float32)
            t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)   # (1, 1, H, W)
            box_preds, cls_preds = model(t)
            boxes_xyxy, scores = decode_predictions(
                box_preds[0], cls_preds[0], model.anchors,
                score_threshold=self._score_threshold,
                class_id=1,
            )
            keep = nms_xyxy(boxes_xyxy, scores, iou_threshold=self._nms_iou)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]

        h_frame, w_frame = X.data.shape
        detections: list[Detection] = []
        for i in range(boxes_xyxy.shape[0]):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            x = max(0.0, x1); y = max(0.0, y1)
            x2 = min(float(w_frame), x2); y2 = min(float(h_frame), y2)
            w = x2 - x; h = y2 - y
            if w <= 0 or h <= 0:
                continue
            detections.append(
                Detection(
                    bbox=(x, y, w, h),
                    score=float(scores[i].item()),
                    class_id=1,
                    camera_id=X.camera_id,
                )
            )
        return detections

    def _validate_shape(self, data: np.ndarray) -> None:
        ew, eh = self.sensor_profile.resolution
        if data.shape != (eh, ew):
            raise ValueError(
                f"Frame shape {data.shape} does not match "
                f"{self.sensor_profile.name} expected (H, W) = ({eh}, {ew})."
            )

    # ---- Persistence ---------------------------------------------------

    def _state_dict(self) -> dict:
        if self._model is None:
            return {}
        return {
            "model_state": self._model.state_dict(),
            "input_size": self._model.input_size,
        }

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        # Rebuild model (CPU first; user can .to('cuda') later via .model).
        import torch
        from thermal_algorithms.human_detection.mobilenet_ssd_model import (
            MicroMobileNetSSD,
        )

        backbone_cfg, ssd_cfg = _pick_configs(self.sensor_profile)
        device = torch.device(self._device_str or "cpu")
        model = MicroMobileNetSSD(
            input_size=tuple(state["input_size"]),
            backbone_config=backbone_cfg,
            ssd_config=ssd_cfg,
        ).to(device)
        # Convert any saved state_dict tensors to the target device.
        sd = {k: v.to(device) if hasattr(v, "to") else v for k, v in state["model_state"].items()}
        model.load_state_dict(sd)
        model.build_anchors(device=device)
        self._model = model
        self._device = device
        self._is_fitted = True
