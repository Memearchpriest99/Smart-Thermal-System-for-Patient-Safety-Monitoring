"""Torch-free inference wrappers used by the demo engine.

- OnnxSSD      — MobileNet-SSD person detector (onnxruntime + numpy decode)
- OnnxContact  — Thermo-X3D T5v2 contact scorer (onnxruntime, rolling buffers)
- FireDetector — FireSVM via the thermal_algorithms package (sklearn)
- BackgroundModel — rolling p25 per-pixel background -> Tateno residual
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# MobileNet-SSD via onnxruntime (numpy anchor decode + NMS)
# ---------------------------------------------------------------------------

class OnnxSSD:
    """Person detector. predict(raw HxW float32) -> list[((x,y,w,h), score)]."""

    def __init__(self, onnx_path: Path, anchors_path: Path, meta_path: Path,
                 threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(
            str(onnx_path), so, providers=["CPUExecutionProvider"])
        self._anchors = np.load(anchors_path)          # (A, 4) cxcywh
        meta = json.loads(Path(meta_path).read_text())
        self._score_th = float(meta["score_threshold"])
        self._nms_iou = float(meta["nms_iou_threshold"])
        self._var_xy = float(meta["bbox_variance_xy"])
        self._var_wh = float(meta["bbox_variance_wh"])
        self._person = int(meta["person_class_id"])
        self._hw = tuple(meta["input_hw"])

    def predict(self, raw: np.ndarray) -> list[tuple[tuple[float, float, float, float], float]]:
        h, w = raw.shape
        x = (raw - raw.mean()) / (raw.std() + 1e-6)
        x = x.astype(np.float32)[None, None]
        box_preds, cls_preds = self._sess.run(None, {"frame": x})
        box_preds, cls_preds = box_preds[0], cls_preds[0]     # (A,4), (A,C)

        # softmax -> person score
        e = np.exp(cls_preds - cls_preds.max(axis=1, keepdims=True))
        scores = (e / e.sum(axis=1, keepdims=True))[:, self._person]
        keep = scores > self._score_th
        if not keep.any():
            return []
        offsets, anchors, scores = box_preds[keep], self._anchors[keep], scores[keep]

        # decode offsets (matches mobilenet_ssd_anchors.decode_boxes)
        cx = offsets[:, 0] * self._var_xy * anchors[:, 2] + anchors[:, 0]
        cy = offsets[:, 1] * self._var_xy * anchors[:, 3] + anchors[:, 1]
        bw = np.exp(offsets[:, 2] * self._var_wh) * anchors[:, 2]
        bh = np.exp(offsets[:, 3] * self._var_wh) * anchors[:, 3]
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2

        keep_idx = self._nms(x1, y1, x2, y2, scores)
        out = []
        for i in keep_idx:
            bx1, by1 = max(0.0, float(x1[i])), max(0.0, float(y1[i]))
            bx2, by2 = min(float(w), float(x2[i])), min(float(h), float(y2[i]))
            if bx2 - bx1 <= 0 or by2 - by1 <= 0:
                continue
            out.append(((bx1, by1, bx2 - bx1, by2 - by1), float(scores[i])))
        return out

    def _nms(self, x1, y1, x2, y2, scores) -> list[int]:
        areas = (x2 - x1) * (y2 - y1)
        order = np.argsort(-scores)
        keep: list[int] = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / np.clip(areas[i] + areas[rest] - inter, 1e-6, None)
            order = rest[iou < self._nms_iou]
        return keep


# ---------------------------------------------------------------------------
# Thermo-X3D contact scorer via onnxruntime
# ---------------------------------------------------------------------------

class OnnxContact:
    """Contact scorer. update(residual_triplet) -> (confidence, alarmed)."""

    def __init__(self, onnx_path: Path, meta_path: Path, threads: int = 4) -> None:
        import onnxruntime as ort

        meta = json.loads(Path(meta_path).read_text())
        self.T = int(meta["T"])
        self._mean = float(meta["global_mean"])
        self._std = float(meta["global_std"])
        self.threshold = float(meta["conf_threshold"])
        self._persistence = int(meta["persistence_frames"])
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self._sess = ort.InferenceSession(
            str(onnx_path), so, providers=["CPUExecutionProvider"])
        self._buffers: list[deque] = [deque(maxlen=self.T) for _ in range(3)]
        self._run = 0

    def reset(self) -> None:
        for b in self._buffers:
            b.clear()
        self._run = 0

    def update(self, residuals: tuple[np.ndarray, np.ndarray, np.ndarray]
               ) -> tuple[Optional[float], bool]:
        for cam, r in enumerate(residuals):
            self._buffers[cam].append(
                ((r - self._mean) / self._std).astype(np.float32))
        if any(len(b) < self.T for b in self._buffers):
            return None, False
        vol = np.stack([np.stack(list(b), 0) for b in self._buffers], 0)[None]
        logits = self._sess.run(None, {"volume": vol})[0][0]
        e = np.exp(logits - logits.max())
        conf = float((e / e.sum())[1])
        self._run = self._run + 1 if conf > self.threshold else 0
        return conf, self._run >= self._persistence


# ---------------------------------------------------------------------------
# FireSVM (sklearn, via thermal_algorithms)
# ---------------------------------------------------------------------------

class FireDetector:
    """predict(raw) -> (is_fire, confidence, bbox or None)."""

    def __init__(self, thalg_path: Path) -> None:
        from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
        from thermal_algorithms.core.types import Frame, FireLevel

        self._det = FireSVMDetector.load(thalg_path)
        self._Frame = Frame
        self._safe = FireLevel.SAFE

    def predict(self, raw: np.ndarray) -> tuple[bool, float, Optional[tuple]]:
        alert = self._det.predict(
            self._Frame(data=raw, timestamp=0.0, camera_id=0))
        is_fire = alert.level != self._safe
        bbox = alert.blob_features.get("bbox") if is_fire else None
        return is_fire, float(alert.confidence), bbox


# ---------------------------------------------------------------------------
# Rolling p25 background -> Tateno residual
# ---------------------------------------------------------------------------

class BackgroundModel:
    """Session-local rolling 25th-percentile background (no calibration).

    Keeps the last `window` raw frames; refreshes the Tateno background
    every `refresh_every` frames. `residual()` returns None until the
    first background is available (warm-up).
    """

    def __init__(self, profile, window: int = 240, min_frames: int = 16,
                 refresh_every: int = 40, bg_pct: float = 25.0) -> None:
        from thermal_algorithms.preprocessing import TatenoPipeline

        self._TatenoPipeline = TatenoPipeline
        self._profile = profile
        self._ring: deque = deque(maxlen=window)
        self._min_frames = min_frames
        self._refresh = refresh_every
        self._pct = bg_pct
        self._count = 0
        self._pipe = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def add(self, raw: np.ndarray) -> None:
        from thermal_algorithms.core.types import Frame

        self._ring.append(raw)
        self._count += 1
        if (self._pipe is None and len(self._ring) >= self._min_frames) or (
                self._pipe is not None and self._count % self._refresh == 0):
            bg = np.percentile(np.stack(self._ring, 0), self._pct, axis=0)
            pipe = self._TatenoPipeline(self._profile).fit(
                [Frame(data=bg.astype(np.float32), timestamp=0.0, camera_id=0)])
            with self._lock:
                self._pipe = pipe

    def residual(self, raw: np.ndarray) -> Optional[np.ndarray]:
        from thermal_algorithms.core.types import Frame

        with self._lock:
            pipe = self._pipe
        if pipe is None:
            return None
        return pipe.predict(
            Frame(data=raw, timestamp=0.0, camera_id=0)).data
