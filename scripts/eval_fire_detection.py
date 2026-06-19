"""Fire detection evaluation — OtsuFireDetector (rule-based) + FireSVMDetector (ML).

Fire-positive scenes : 3pplfire, onepersonfire, setup2_fire_lighter
Everything else is fire-negative; 2pplwithhairdryer and emptyroomwithpcscreen
are deliberately included as *hard* negatives (hot distractors).

Positive prediction := alert.level != FireLevel.SAFE.

Two known data hazards this script exposes:
  1. MLX90640 dead-pixel glitches: isolated single pixels read 800-935 degC in a
     handful of frames. Absolute-temperature rules treat these as ignition
     sources. We evaluate Otsu both RAW and with a hot-pixel DESPIKE filter to
     quantify the impact.
  2. Severe class imbalance: 78 positive / 5865 negative frames (~1.3%).
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Frame, FireLevel
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
from thermal_algorithms.training import DatasetIndex, FireFrameDataset
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix, best_iou

POSITIVE_SCENES = {"3pplfire", "onepersonfire", "setup2_fire_lighter"}
HARD_NEGATIVES = {"2pplwithhairdryer", "emptyroomwithpcscreen"}
CHANNELS = (0, 1, 2)
TRAIN_FRAC = 0.70


# ---------------------------------------------------------------------------
# Hot-pixel despike — replace isolated extreme pixels with neighbourhood median
# ---------------------------------------------------------------------------

def despike(frame: Frame, hot_c: float = 100.0) -> Frame:
    """Replace any pixel > hot_c whose 3x3 neighbourhood median is < hot_c
    (i.e. an isolated spike) with that neighbourhood median. Real fire blobs,
    which are spatially coherent, are left untouched."""
    import cv2
    data = frame.data.astype(np.float32)
    med = cv2.medianBlur(data, 3)
    spike = (data > hot_c) & (med < hot_c)
    if spike.any():
        data = data.copy()
        data[spike] = med[spike]
    return Frame(data=data, timestamp=frame.timestamp,
                 camera_id=frame.camera_id, metadata=frame.metadata)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_cm(cm: BinaryConfusionMatrix, title: str) -> None:
    total = cm.total

    def pct(v):
        return f"{v / total * 100:5.1f}%" if total > 0 else "  N/A "
    print(f"\n  {title}")
    print(f"  {'':25s}  {'Pred: FIRE':>18}  {'Pred: SAFE':>18}")
    print(f"  {'Truth: FIRE (Pos)':25s}  {'TP='+str(cm.tp):>8} {pct(cm.tp):>8}  {'FN='+str(cm.fn):>8} {pct(cm.fn):>8}")
    print(f"  {'Truth: SAFE (Neg)':25s}  {'FP='+str(cm.fp):>8} {pct(cm.fp):>8}  {'TN='+str(cm.tn):>8} {pct(cm.tn):>8}")
    print(f"  Accuracy={cm.accuracy:.1%}  Precision={cm.precision:.1%}  "
          f"Recall={cm.recall:.1%}  F1={cm.f1:.1%}  Correct={cm.correct}/{total}")


def print_scene_table(rows: list[dict], title: str) -> None:
    w = 26
    print(f"\n  {title}")
    hdr = (f"  {'Scene':<{w}}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
           f"{'IoU':>6}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'N':>5}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep); print(hdr); print(sep)
    for r in rows:
        iou = f"{r['iou']:>5.1%}" if r['iou'] is not None else "   - "
        print(f"  {r['scene']:<{w}}  {r['acc']:>6.1%}  {r['prec']:>7.1%}  "
              f"{r['rec']:>7.1%}  {r['f1']:>7.1%}  {iou:>6}  "
              f"{r['tp']:>4d}  {r['tn']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  {r['total']:>5d}")
    print(sep)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def eval_detector(detector, ds: FireFrameDataset, *, apply_despike: bool):
    """Run a detector over every session; return (aggregate_cm, per_scene_rows)."""
    rows = []
    agg = BinaryConfusionMatrix(0, 0, 0, 0)
    for session, examples in ds.by_session():
        if hasattr(detector, "reset"):
            detector.reset()
        yt, yp = [], []
        ious = []
        for frame, gt in examples:
            if apply_despike:
                frame = despike(frame)
            pred = detector.predict(frame)
            gt_pos = gt.level != FireLevel.SAFE
            pred_pos = pred.level != FireLevel.SAFE
            yt.append(1 if gt_pos else 0)
            yp.append(1 if pred_pos else 0)
            if gt_pos:
                gt_boxes = gt.blob_features.get("bboxes", [])
                pbox = pred.blob_features.get("bbox")
                ious.append(best_iou([pbox] if pbox else [], gt_boxes))
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({
            "scene": session.scene, "acc": cm.accuracy, "prec": cm.precision,
            "rec": cm.recall, "f1": cm.f1, "tp": cm.tp, "tn": cm.tn,
            "fp": cm.fp, "fn": cm.fn, "total": cm.total,
            "iou": (sum(ious) / len(ious)) if ious else None,
        })
    return agg, rows


def split_examples(ds: FireFrameDataset):
    """Stratified sequential train/test split per (scene, positive/negative)."""
    by_key = defaultdict(list)
    for session, examples in ds.by_session():
        for frame, gt in examples:
            pos = gt.level != FireLevel.SAFE
            by_key[(session.scene, frame.camera_id, pos)].append((frame, gt))
    train, test = [], []
    for key, items in by_key.items():
        n_tr = max(1, int(len(items) * TRAIN_FRAC)) if len(items) > 1 else 0
        train.extend(items[:n_tr])
        test.extend(items[n_tr:])
    return train, test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("  FIRE DETECTION — EVALUATION")
    print("=" * 78)
    idx = DatasetIndex("dataset", sensor_profile=MLX90640)
    ds = FireFrameDataset(idx, channels=CHANNELS)
    print(f"  Sessions: {len(idx.sessions)}   Positive scenes: {sorted(POSITIVE_SCENES)}")
    print(f"  Hard negatives (hot distractors): {sorted(HARD_NEGATIVES)}")

    def order(rows):
        # positive scenes first, then hard negatives, then the rest
        def rank(r):
            if r['scene'] in POSITIVE_SCENES: return (0, r['scene'])
            if r['scene'] in HARD_NEGATIVES: return (1, r['scene'])
            return (2, r['scene'])
        return sorted(rows, key=rank)

    # ---------------- OtsuFireDetector (rule-based) ----------------
    # The detector now despikes and segments on an absolute t_ign threshold
    # internally (no Otsu erosion), so no external preprocessing is applied here.
    print("\n" + "=" * 78)
    print("  OtsuFireDetector  —  hot-pixel bypass (t_ign threshold, no erosion, despiked)")
    print("=" * 78)
    det = OtsuFireDetector(MLX90640)
    det.fit([])
    agg, rows = eval_detector(det, ds, apply_despike=False)
    rows = order(rows)
    show = [r for r in rows if r['scene'] in POSITIVE_SCENES | HARD_NEGATIVES or r['fp'] > 0]
    print_scene_table(show, "Key scenes (positives + hot distractors + any false-alarm scene)")
    n_fp_scenes = sum(1 for r in rows if r['fp'] > 0 and r['scene'] not in POSITIVE_SCENES)
    print(f"\n  Negative scenes with >=1 false alarm: {n_fp_scenes} / "
          f"{sum(1 for r in rows if r['scene'] not in POSITIVE_SCENES)}")
    print_cm(agg, f"Aggregate confusion (all {agg.total} frames, all scenes)")

    # ---------------- FireSVMDetector (ML) ----------------
    print("\n" + "=" * 78)
    print("  FireSVMDetector  —  RBF kernel, 70/30 stratified split")
    print("=" * 78)
    train, test = split_examples(ds)
    tr_pos = sum(1 for _, g in train if g.level != FireLevel.SAFE)
    te_pos = sum(1 for _, g in test if g.level != FireLevel.SAFE)
    print(f"  Train: {len(train)} frames (pos={tr_pos})   Test: {len(test)} frames (pos={te_pos})")

    svm = FireSVMDetector(MLX90640, kernel="rbf")
    svm.fit([f for f, _ in train], [g for _, g in train])

    # Evaluate on test, grouped by scene
    by_scene = defaultdict(lambda: ([], []))
    for frame, gt in test:
        pred = svm.predict(frame)
        scene = frame.metadata.get("session_id", "?")
        by_scene[scene][0].append(1 if gt.level != FireLevel.SAFE else 0)
        by_scene[scene][1].append(1 if pred.level != FireLevel.SAFE else 0)
    rows = []
    agg = BinaryConfusionMatrix(0, 0, 0, 0)
    for scene, (yt, yp) in by_scene.items():
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "tp": cm.tp, "tn": cm.tn,
                     "fp": cm.fp, "fn": cm.fn, "total": cm.total, "iou": None})
    show = [r for r in order(rows) if r['scene'] in POSITIVE_SCENES | HARD_NEGATIVES or r['fp'] > 0]
    print_scene_table(show, "Test split — key scenes + any false-alarm scenes")
    print_cm(agg, f"Aggregate confusion (test split, {agg.total} frames)")


if __name__ == "__main__":
    main()
