"""Evaluation metrics matching the §5.3 Algorithmic Performance Evaluation tables.

Three metric families are implemented:

1. **Classification metrics** (§5.3.2, §5.3.4)
   Accuracy, Precision, Recall, F1-Score, False-Alarm Rate — derived from a
   ``BinaryConfusionMatrix`` built by reducing each frame to a binary
   "detected / not detected" verdict.  Matches Table 3 (Human Detection) and
   Table 4 (Fire Detection) in the Engineering Report exactly.

2. **Bounding-box / region IoU** (§5.3.4)
   Per-frame Intersection-over-Union between predicted regions (bbox or binary
   mask) and ground-truth bounding boxes.  Both the PASCAL-VOC bbox-vs-bbox
   and the segmentation mask-vs-bbox variants are provided because the Otsu
   pipeline output is a pixel mask, not a box.  Matches the IoU column in
   Table 4.

3. **Signal-to-Background Ratio** (§5.3.1)
   SBR = μ_signal / μ_background, computed from the target region (inside
   ground-truth bounding boxes) versus the rest of the frame.  The report
   reports 5.61× improvement (raw 1.21 → processed 6.80) using this metric.

All functions are pure numpy — no framework dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ===========================================================================
# 1. Binary confusion matrix and derived classification metrics
# ===========================================================================

@dataclass
class BinaryConfusionMatrix:
    """Four-cell confusion matrix for binary (Positive / Negative) evaluation.

    Attribute names follow the conventions used in the Engineering Report:
        Positive  = "Human present" / "Fire present" / "Contact detected"
        Negative  = "Empty room" / "No fire" / "No contact"

    Attributes:
        tp: True Positives  — frame predicted Positive, ground truth Positive.
        tn: True Negatives  — frame predicted Negative, ground truth Negative.
        fp: False Positives — frame predicted Positive, ground truth Negative.
        fn: False Negatives — frame predicted Negative, ground truth Positive.
    """
    tp: int
    tn: int
    fp: int
    fn: int

    # ---- Derived metrics -----------------------------------------------

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def correct(self) -> int:
        """Count of correctly classified frames (TP + TN)."""
        return self.tp + self.tn

    @property
    def accuracy(self) -> float:
        """(TP + TN) / total.  Returns 0 for empty inputs."""
        t = self.total
        return self.correct / t if t > 0 else 0.0

    @property
    def precision(self) -> float:
        """TP / (TP + FP).  Returns 0 when no positive predictions are made."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN).  Returns 0 when no actual positives exist.

        Recall is the primary metric for safety systems: a value below 1.0
        implies a missed fire / contact event.
        """
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """2 · Precision · Recall / (Precision + Recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_alarm_rate(self) -> float:
        """FP / (FP + TN).  Also called fall-out or 1 − specificity.

        Critical for a deployed safety system: excessive false alarms cause
        alert fatigue, leading staff to disable or ignore the system.
        """
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0

    # ---- Combination -------------------------------------------------------

    def __add__(self, other: "BinaryConfusionMatrix") -> "BinaryConfusionMatrix":
        return BinaryConfusionMatrix(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )

    # ---- Formatting --------------------------------------------------------

    def as_matrix(self) -> np.ndarray:
        """Return a 2×2 array [[TN, FP], [FN, TP]] (sklearn convention)."""
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f"BinaryConfusionMatrix("
            f"acc={self.accuracy:.1%}, prec={self.precision:.1%}, "
            f"rec={self.recall:.1%}, f1={self.f1:.1%}, "
            f"correct={self.correct}/{self.total})"
        )


def binary_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> BinaryConfusionMatrix:
    """Build a BinaryConfusionMatrix from paired label sequences.

    Args:
        y_true: Ground-truth binary labels (1 = Positive, 0 = Negative).
        y_pred: Predicted binary labels (1 = Positive, 0 = Negative).

    Returns:
        Populated ``BinaryConfusionMatrix``.

    Raises:
        ValueError: If lengths differ or arrays contain values other than 0/1.
    """
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {len(yt)} vs {len(yp)}."
        )
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    return BinaryConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)


def detections_to_binary_labels(
    detections_per_frame: list[list],
    gt_per_frame: list[list],
) -> tuple[list[int], list[int]]:
    """Convert per-frame detection lists to binary presence labels.

    A frame is predicted Positive if the detector returns at least one
    detection.  A frame is ground-truth Positive if it has at least one
    annotated bbox.

    This is the exact reduction used to build Tables 3 and 4 in the report.

    Args:
        detections_per_frame: Parallel lists of predicted Detection objects.
        gt_per_frame: Parallel lists of ground-truth Detection objects.

    Returns:
        (y_pred, y_true) as parallel integer lists.
    """
    y_pred = [1 if dets else 0 for dets in detections_per_frame]
    y_true = [1 if gts else 0 for gts in gt_per_frame]
    return y_pred, y_true


# ===========================================================================
# 2.  Bounding-box and region IoU
# ===========================================================================

def iou_bbox(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
) -> float:
    """Intersection-over-Union between two bounding boxes.

    Both boxes use (x, y, w, h) top-left + size convention, matching
    ``Detection.bbox`` throughout the library.

    Returns 0.0 when either box has zero area.
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    ix1 = max(px, gx)
    iy1 = max(py, gy)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    pred_area = pw * ph
    gt_area = gw * gh
    union = pred_area + gt_area - inter
    return float(inter / union) if union > 0 else 0.0


def iou_mask_vs_bbox(
    mask: np.ndarray,
    gt_bbox: tuple[float, float, float, float],
) -> float:
    """IoU between a binary segmentation mask and a ground-truth bounding box.

    Used for fire detection evaluation (§5.3.4): the Otsu pipeline produces
    a binary mask; the ground truth is an annotated bounding box around the
    flame.  The mask is compared pixel-by-pixel with the filled gt rectangle.

    Args:
        mask: 2-D binary array (non-zero = foreground), same shape as the
            source thermal frame.
        gt_bbox: (x, y, w, h) ground-truth box in pixel coordinates.

    Returns:
        IoU ∈ [0, 1].  Returns 0 when either region is empty.
    """
    h, w = mask.shape
    gt_mask = np.zeros((h, w), dtype=bool)
    gx, gy, gw, gh = gt_bbox
    x0 = max(0, int(round(gx)))
    y0 = max(0, int(round(gy)))
    x1 = min(w, int(round(gx + gw)))
    y1 = min(h, int(round(gy + gh)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    gt_mask[y0:y1, x0:x1] = True

    pred_mask = mask.astype(bool)
    inter = int((pred_mask & gt_mask).sum())
    union = int((pred_mask | gt_mask).sum())
    return float(inter / union) if union > 0 else 0.0


def best_iou(
    pred_bboxes: list[tuple[float, float, float, float]],
    gt_bboxes: list[tuple[float, float, float, float]],
) -> float:
    """Maximum IoU of any predicted box against any ground-truth box.

    Returns 0.0 when either list is empty.
    """
    if not pred_bboxes or not gt_bboxes:
        return 0.0
    return max(iou_bbox(p, g) for p in pred_bboxes for g in gt_bboxes)


def mean_detection_iou(
    pred_bboxes_per_frame: list[list[tuple[float, float, float, float]]],
    gt_bboxes_per_frame: list[list[tuple[float, float, float, float]]],
) -> float:
    """Mean best-IoU over frames where the ground truth is non-empty.

    Frames with no ground-truth annotations are excluded from the average
    (they cannot contribute a meaningful IoU value).

    This matches the Mean IoU column in Table 4 of the Engineering Report.
    """
    scores: list[float] = []
    for preds, gts in zip(pred_bboxes_per_frame, gt_bboxes_per_frame):
        if gts:
            scores.append(best_iou(preds, gts))
    return float(np.mean(scores)) if scores else 0.0


# ===========================================================================
# 3.  Signal-to-Background Ratio  (§5.3.1)
# ===========================================================================

def signal_to_background_ratio(
    frame_data: np.ndarray,
    gt_bboxes: list[tuple[float, float, float, float]],
) -> float:
    """Compute the Signal-to-Background Ratio for one thermal frame.

    SBR = μ_signal / μ_background

    where:
        μ_signal     = mean temperature of all pixels inside any gt bbox.
        μ_background = mean temperature of all pixels outside every gt bbox.

    The Engineering Report achieved 5.61× improvement (1.21 → 6.80) for the
    Tateno preprocessing pipeline using this metric.

    Args:
        frame_data: 2-D float thermal array (H, W).
        gt_bboxes: Ground-truth bounding boxes in (x, y, w, h) pixel coords.
            If empty, returns 1.0 (no signal region defined).

    Returns:
        SBR ≥ 0.  Returns 0.0 if background mean is zero or negative.
    """
    if not gt_bboxes:
        return 1.0

    h, w = frame_data.shape
    signal_mask = np.zeros((h, w), dtype=bool)
    for gx, gy, gw, gh in gt_bboxes:
        x0 = max(0, int(round(gx)))
        y0 = max(0, int(round(gy)))
        x1 = min(w, int(round(gx + gw)))
        y1 = min(h, int(round(gy + gh)))
        if x1 > x0 and y1 > y0:
            signal_mask[y0:y1, x0:x1] = True

    signal_pixels = frame_data[signal_mask].astype(np.float64)
    bg_pixels = frame_data[~signal_mask].astype(np.float64)

    if signal_pixels.size == 0 or bg_pixels.size == 0:
        return 1.0

    mu_signal = float(signal_pixels.mean())
    mu_bg = float(bg_pixels.mean())

    if mu_bg <= 0:
        return 0.0
    return mu_signal / mu_bg


def mean_sbr(
    frames: list[np.ndarray],
    gt_bboxes_per_frame: list[list[tuple[float, float, float, float]]],
) -> float:
    """Mean SBR across a sequence of frames."""
    ratios = [
        signal_to_background_ratio(f, b)
        for f, b in zip(frames, gt_bboxes_per_frame)
    ]
    return float(np.mean(ratios)) if ratios else 1.0


# ===========================================================================
# 4.  Scenario-level evaluation  (matches Tables 3 & 4 structure)
# ===========================================================================

@dataclass
class ScenarioResult:
    """Per-scenario evaluation result matching the structure of Tables 3 & 4.

    Attributes:
        name:       Folder / scenario name (e.g. '1_man_out_in_out_7_sec').
        mode:       'raw' or 'proc' (raw sensor data vs. preprocessed).
        confusion:  Binary confusion matrix.
        mean_iou:   Mean IoU over positive frames (0.0 if not applicable).
    """
    name: str
    mode: str
    confusion: BinaryConfusionMatrix
    mean_iou: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.confusion.accuracy

    @property
    def precision(self) -> float:
        return self.confusion.precision

    @property
    def recall(self) -> float:
        return self.confusion.recall

    @property
    def f1(self) -> float:
        return self.confusion.f1

    @property
    def correct_over_total(self) -> str:
        return f"{self.confusion.correct}/{self.confusion.total}"


def format_scenario_table(
    results: list[ScenarioResult],
    *,
    include_iou: bool = False,
) -> str:
    """Format a list of ScenarioResults into a human-readable ASCII table.

    Reproduces the layout of Tables 3 and 4 in the Engineering Report.

    Args:
        results: Scenario results to include; multiple modes for the same
            scenario name are printed on consecutive rows (like the report).
        include_iou: Set True for fire-detection tables (adds IoU column).

    Returns:
        Formatted multi-line string.
    """
    iou_col = "  IoU  " if include_iou else ""
    header = (
        f"{'Folder Name':<40} {'Mode':<5}  {iou_col}"
        f"{'Accuracy':>9}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  "
        f"Correct/Total"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        iou_val = f"{r.mean_iou:>6.1%}  " if include_iou else ""
        lines.append(
            f"{r.name:<40} {r.mode:<5}  {iou_val}"
            f"{r.accuracy:>9.1%}  {r.precision:>9.1%}  {r.recall:>7.1%}  "
            f"{r.f1:>7.1%}  {r.correct_over_total}"
        )
    lines.append(sep)
    return "\n".join(lines)


def aggregate_scenarios(
    results: list[ScenarioResult],
    mode: Optional[str] = None,
) -> BinaryConfusionMatrix:
    """Sum confusion matrices across all (or a filtered subset of) scenarios.

    Args:
        results: All scenario results.
        mode: If given, only results with this mode string are included.

    Returns:
        Aggregated ``BinaryConfusionMatrix`` for macro-level reporting.
    """
    filtered = [r for r in results if mode is None or r.mode == mode]
    if not filtered:
        return BinaryConfusionMatrix(0, 0, 0, 0)
    agg = filtered[0].confusion
    for r in filtered[1:]:
        agg = agg + r.confusion
    return agg
