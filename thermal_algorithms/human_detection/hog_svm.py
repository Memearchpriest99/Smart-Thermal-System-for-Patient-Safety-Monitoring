"""HOGSVMDetector - HOG + Linear SVM human detection (Section 4.4.2.2).

The detector treats human detection as shape analysis rather than intensity
analysis: it computes Histogram-of-Oriented-Gradients features at every
sliding-window position in the frame and asks a trained Linear SVM whether
that window looks like a person.

Pipeline (matching the report's "Mechanism"):

    1. Gradient computation     - per-pixel horizontal and vertical gradients
                                  (provided by skimage.feature.hog).
    2. Cell histogramming       - bin gradient magnitudes by orientation in
                                  cells of `cell_size`.
    3. Block normalization      - L2 normalize histograms across blocks of
                                  `block_size` cells (handles local contrast).
    4. SVM classification       - linear hyperplane in HOG feature space.

The trained system implements:

    f(X) = sign(W . X + b)

Where X is the HOG feature vector for one window. The decision margin
`W . X + b` is also passed through a sigmoid to populate
`Detection.score` with a quasi-probability in (0, 1).

Resolution behavior
-------------------
'parameterized' with per-profile checkpoints. The canonical detection
window is sized from a *physical* person silhouette (default 1.7 m tall x
0.4 m wide), converted to pixels via the SensorProfile. The cell size is
auto-derived to give roughly the same number of cells in the window across
sensors, so the HOG feature dimension is comparable across the MLX and
Waveshare paths (but checkpoints still don't transfer across them).
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.human_detection.base import HumanDetector


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def _greedy_nms(
    boxes: list[tuple[float, float, float, float]],
    scores: list[float],
    iou_threshold: float,
) -> list[int]:
    """Greedy NMS over (x, y, w, h) boxes. Returns indices to keep."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    x1, y1 = arr[:, 0], arr[:, 1]
    x2, y2 = x1 + arr[:, 2], y1 + arr[:, 3]
    areas = arr[:, 2] * arr[:, 3]
    order = np.argsort(scores)[::-1]

    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_threshold]
    return keep


class HOGSVMDetector(HumanDetector):
    """HOG + Linear SVM human detector (Section 4.4.2.2)."""

    name = "hog_svm_detector"
    is_trainable = True
    resolution_behavior = "parameterized"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        # Window
        window_size: Optional[tuple[int, int]] = None,   # (H, W) in pixels
        physical_person_size_m: tuple[float, float] = (0.9, 0.45),
        assumed_distance_m: float = 2.0,
        max_window_fraction: float = 0.65,
        # HOG
        cell_size: Optional[tuple[int, int]] = None,     # (cell_h, cell_w)
        block_size_cells: tuple[int, int] = (2, 2),      # block extent in cells
        orientations: int = 9,
        # SVM
        svm_C: float = 1.0,
        # Inference
        stride: tuple[int, int] = (1, 1),
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        pyramid_scales: tuple[float, ...] = (1.0,),
        # Negative sampling defaults (when caller doesn't provide negatives)
        n_negatives_per_frame: int = 5,
        random_state: int = 0,
        class_weight: Optional[str | dict] = None,
    ) -> None:
        """
        class_weight: Passed straight through to sklearn's ``LinearSVC`` —
            ``'balanced'`` or an explicit ``{0: w0, 1: w1}`` dict. A
            constructor parameter (not `fit()`-time) because sklearn's
            `class_weight` is fixed at estimator construction.
        """
        if sensor_profile is None:
            raise ValueError("HOGSVMDetector requires a SensorProfile.")

        super().__init__(
            sensor_profile=sensor_profile,
            window_size=window_size,
            physical_person_size_m=physical_person_size_m,
            assumed_distance_m=assumed_distance_m,
            max_window_fraction=max_window_fraction,
            cell_size=cell_size,
            block_size_cells=block_size_cells,
            orientations=orientations,
            svm_C=svm_C,
            stride=stride,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
            pyramid_scales=pyramid_scales,
            n_negatives_per_frame=n_negatives_per_frame,
            random_state=random_state,
            class_weight=class_weight,
        )
        self._class_weight = class_weight

        self._cell_size = self._resolve_cell_size(window_size or (0, 0), cell_size)
        resolved_wh_ww = self._resolve_window_size(
            sensor_profile, window_size, physical_person_size_m, assumed_distance_m, max_window_fraction
        )
        self._window_size = self._snap_to_multiple(resolved_wh_ww, self._cell_size)
        self._block_size_cells = tuple(int(v) for v in block_size_cells)
        self._orientations = int(orientations)
        self._svm_C = float(svm_C)
        self._stride = (int(stride[0]), int(stride[1]))
        self._score_threshold = float(score_threshold)
        self._nms_iou = float(nms_iou_threshold)
        if not pyramid_scales:
            raise ValueError("pyramid_scales must contain at least one scale.")
        self._pyramid_scales = [float(s) for s in pyramid_scales]
        self._n_negatives_per_frame = int(n_negatives_per_frame)
        self._rng = np.random.default_rng(int(random_state))

        # Validate cell + window divisibility for HOG.
        wh, ww = self._window_size
        ch, cw = self._cell_size
        if wh % ch or ww % cw:
            raise ValueError(
                f"window_size {self._window_size} must be divisible by "
                f"cell_size {self._cell_size}."
            )

        # Learned state - populated by fit().
        self._svm: Optional[LinearSVC] = None
        self._n_features: Optional[int] = None

    # ---- Resolver helpers -------------------------------------------------

    @staticmethod
    def _resolve_window_size(
        profile: SensorProfile,
        window_size: Optional[tuple[int, int]],
        physical_person_size_m: tuple[float, float],
        assumed_distance_m: float,
        max_window_fraction: float,
    ) -> tuple[int, int]:
        if not 0 < max_window_fraction <= 1.0:
            raise ValueError(f"max_window_fraction must be in (0, 1]; got {max_window_fraction}.")
        fw, fh = profile.resolution
        max_wh = max(4, int(fh * max_window_fraction))
        max_ww = max(2, int(fw * max_window_fraction))

        if window_size is not None:
            wh = min(int(window_size[0]), max_wh)
            ww = min(int(window_size[1]), max_ww)
            return wh, ww
        if any(v <= 0 for v in physical_person_size_m) or assumed_distance_m <= 0:
            raise ValueError("physical_person_size_m components and distance must be positive.")
        dx, dy = profile.physical_pixel_size_m(assumed_distance_m)
        person_h_m, person_w_m = physical_person_size_m
        wh = max(4, int(round(person_h_m / dy)))
        ww = max(2, int(round(person_w_m / dx)))
        # Cap against the frame so a sliding window can actually slide.
        wh = min(wh, max_wh)
        ww = min(ww, max_ww)
        return wh, ww

    @staticmethod
    def _resolve_cell_size(
        window_size: tuple[int, int],
        cell_size: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        """Cell size in pixels. Default (2, 2) is standard for small thermal
        windows where a single pixel is a meaningful spatial element."""
        if cell_size is not None:
            return (int(cell_size[0]), int(cell_size[1]))
        return (2, 2)

    @staticmethod
    def _snap_to_multiple(window_size: tuple[int, int], cell_size: tuple[int, int]) -> tuple[int, int]:
        """Snap window dims DOWN to the nearest multiple of cell_size so HOG
        gets an integer number of cells. Guarantees the window has at least
        2 cells along each axis so a 2x2 block fits."""
        wh, ww = window_size
        ch, cw = cell_size
        # Snap down.
        wh -= wh % ch
        ww -= ww % cw
        wh = max(wh, 2 * ch)
        ww = max(ww, 2 * cw)
        return wh, ww

    # ---- Read-only resolved parameters -----------------------------------

    @property
    def window_size(self) -> tuple[int, int]:
        return self._window_size

    @property
    def cell_size(self) -> tuple[int, int]:
        return self._cell_size

    @property
    def feature_dim(self) -> Optional[int]:
        return self._n_features

    # ---- Internal building blocks ----------------------------------------

    def _validate_shape(self, data: np.ndarray) -> None:
        ew, eh = self.sensor_profile.resolution
        if data.shape != (eh, ew):
            raise ValueError(
                f"Frame shape {data.shape} does not match "
                f"{self.sensor_profile.name} expected (H, W) = ({eh}, {ew})."
            )

    def _normalize_patch(self, patch: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance per window. Makes the detector robust to
        the ambient temperature of the scene without needing to know it."""
        p = patch.astype(np.float32)
        mu = p.mean()
        sigma = p.std()
        if sigma < 1e-6:
            return p - mu
        return (p - mu) / sigma

    def _hog_features(self, patch: np.ndarray) -> np.ndarray:
        """Compute a HOG feature vector for one normalized window-sized patch."""
        return hog(
            patch,
            orientations=self._orientations,
            pixels_per_cell=self._cell_size,
            cells_per_block=self._block_size_cells,
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

    def _crop_resize(self, frame_data: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        x, y, w, h = bbox
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(frame_data.shape[1], int(round(x + w)))
        y1 = min(frame_data.shape[0], int(round(y + h)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Degenerate bbox after clamping: {bbox}")
        crop = frame_data[y0:y1, x0:x1]
        wh, ww = self._window_size
        return cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_AREA)

    def _sample_negatives(
        self, frame_data: np.ndarray, positives: list[Detection],
    ) -> list[np.ndarray]:
        """Random non-overlapping background patches from a labeled frame."""
        wh, ww = self._window_size
        H, W = frame_data.shape
        if H < wh or W < ww:
            return []
        boxes = np.array(
            [(d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3])
             for d in positives],
            dtype=np.float32,
        ) if positives else np.zeros((0, 4), dtype=np.float32)

        patches: list[np.ndarray] = []
        max_attempts = self._n_negatives_per_frame * 10
        attempts = 0
        while len(patches) < self._n_negatives_per_frame and attempts < max_attempts:
            attempts += 1
            x0 = int(self._rng.integers(0, W - ww + 1))
            y0 = int(self._rng.integers(0, H - wh + 1))
            x1, y1 = x0 + ww, y0 + wh
            # Check IoU against all positives.
            keep = True
            for bx0, by0, bx1, by1 in boxes:
                ix0 = max(x0, bx0)
                iy0 = max(y0, by0)
                ix1 = min(x1, bx1)
                iy1 = min(y1, by1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    union = (wh * ww) + (bx1 - bx0) * (by1 - by0) - inter
                    if union > 0 and inter / union > 0.1:
                        keep = False
                        break
            if keep:
                patches.append(frame_data[y0:y1, x0:x1])
        return patches

    # ---- Public API ------------------------------------------------------

    def fit(
        self,
        X: Iterable,
        y: None = None,
        *,
        negatives: Optional[np.ndarray] = None,
    ) -> "HOGSVMDetector":
        """Train the SVM on labeled frames.

        Args:
            X: Iterable yielding (Frame, list[Detection]) tuples.
            y: Unused (X already carries labels).
            negatives: Optional (N, H, W) array of explicit background patches.
                If provided, used INSTEAD of internal random negative sampling.
        """
        pos_feats: list[np.ndarray] = []
        neg_feats: list[np.ndarray] = []

        for item in X:
            frame, dets = item
            self._validate_shape(frame.data)

            # Positives: each labeled bbox in this frame.
            for d in dets:
                try:
                    patch = self._crop_resize(frame.data, d.bbox)
                except ValueError:
                    continue
                pos_feats.append(self._hog_features(self._normalize_patch(patch)))

            # Internal negatives if user didn't provide explicit ones.
            if negatives is None:
                for patch in self._sample_negatives(frame.data, dets):
                    neg_feats.append(self._hog_features(self._normalize_patch(patch)))

        if negatives is not None:
            wh, ww = self._window_size
            for k in range(negatives.shape[0]):
                patch = negatives[k]
                if patch.shape != (wh, ww):
                    patch = cv2.resize(patch, (ww, wh), interpolation=cv2.INTER_AREA)
                neg_feats.append(self._hog_features(self._normalize_patch(patch)))

        if not pos_feats:
            raise ValueError("fit() received no positive examples - check class_filter and dataset.")
        if not neg_feats:
            raise ValueError("fit() received no negative examples - provide negatives= or labeled frames with background.")

        X_mat = np.vstack(pos_feats + neg_feats)
        y_vec = np.concatenate([
            np.ones(len(pos_feats), dtype=np.int32),
            np.zeros(len(neg_feats), dtype=np.int32),
        ])

        self._svm = LinearSVC(
            C=self._svm_C, dual="auto", max_iter=5000, class_weight=self._class_weight,
        )
        self._svm.fit(X_mat, y_vec)
        self._n_features = int(X_mat.shape[1])
        self._is_fitted = True
        return self

    def _predict_at_scale(
        self,
        frame_data: np.ndarray,
        scale: float,
    ) -> tuple[list, list, list]:
        """Slide the detection window over a single rescaled copy of the frame.

        Returns (bboxes, scores, features) in original-frame coordinates.
        """
        if abs(scale - 1.0) < 1e-6:
            scaled = frame_data
        else:
            H, W = frame_data.shape
            new_h = max(self._window_size[0], int(round(H * scale)))
            new_w = max(self._window_size[1], int(round(W * scale)))
            interp = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            scaled = cv2.resize(frame_data, (new_w, new_h), interpolation=interp)

        wh, ww = self._window_size
        sh, sw = self._stride
        H_s, W_s = scaled.shape

        bboxes: list[tuple[float, float, float, float]] = []
        scores: list[float] = []
        feats: list[dict] = []

        for y0 in range(0, H_s - wh + 1, sh):
            for x0 in range(0, W_s - ww + 1, sw):
                patch = scaled[y0:y0 + wh, x0:x0 + ww]
                features = self._hog_features(self._normalize_patch(patch))
                margin = float(self._svm.decision_function(features.reshape(1, -1))[0])
                if margin < self._score_threshold:
                    continue
                bboxes.append((x0 / scale, y0 / scale, ww / scale, wh / scale))
                scores.append(margin)
                feats.append({
                    "svm_margin": margin,
                    "pyramid_scale": scale,
                    "max_temp": float(patch.max()),
                    "mean_temp": float(patch.mean()),
                    "std_temp": float(patch.std()),
                })
        return bboxes, scores, feats

    def predict(self, X: Frame) -> list[Detection]:
        if self._svm is None or not self._is_fitted:
            raise RuntimeError("HOGSVMDetector.predict() called before fit().")
        self._validate_shape(X.data)

        all_bbox: list[tuple[float, float, float, float]] = []
        all_score: list[float] = []
        all_feats: list[dict] = []

        for scale in self._pyramid_scales:
            bb, sc, ft = self._predict_at_scale(X.data, scale)
            all_bbox.extend(bb)
            all_score.extend(sc)
            all_feats.extend(ft)

        keep = _greedy_nms(all_bbox, all_score, self._nms_iou)

        return [
            Detection(
                bbox=all_bbox[i],
                score=float(_sigmoid(all_score[i])),
                class_id=1,
                camera_id=X.camera_id,
                thermal_features=all_feats[i],
            )
            for i in keep
        ]

    # ---- Persistence -----------------------------------------------------

    def _state_dict(self) -> dict:
        if self._svm is None:
            return {}
        return {
            "svm_coef": self._svm.coef_,
            "svm_intercept": self._svm.intercept_,
            "svm_classes": self._svm.classes_,
            "n_features": self._n_features,
            "resolved_window_size": self._window_size,
            "resolved_cell_size": self._cell_size,
        }

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        # Reconstruct a minimal LinearSVC with the learned weights.
        svm = LinearSVC(C=self._svm_C, dual="auto", max_iter=5000)
        svm.coef_ = np.asarray(state["svm_coef"], dtype=np.float64)
        svm.intercept_ = np.asarray(state["svm_intercept"], dtype=np.float64)
        svm.classes_ = np.asarray(state["svm_classes"])
        # sklearn requires these private attrs to allow decision_function.
        svm.n_features_in_ = int(state["n_features"])
        self._svm = svm
        self._n_features = int(state["n_features"])
        if "resolved_window_size" in state:
            self._window_size = tuple(int(v) for v in state["resolved_window_size"])
        if "resolved_cell_size" in state:
            self._cell_size = tuple(int(v) for v in state["resolved_cell_size"])
        self._is_fitted = True
