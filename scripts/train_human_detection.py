"""Human detection training and evaluation — 80/10/10 train/val/test split.

Evaluates all three detectors:
  1. AdaptiveThresholdDetector  (rule-based, no training)
  2. HOGSVMDetector             (HOG + Linear SVM, classical ML)
  3. MobileNetSSDDetector       (deep, PyTorch — auto-skipped if torch missing)

Split strategy
--------------
Within each (scene, channel) group, labeled frame indices are sorted and
split sequentially: first 80% → train, next 10% → val, last 10% → test.
Sequential splitting avoids temporal leakage while ensuring every scene
contributes to every split proportionally.

For the empty room scene, even-indexed frames are used to calibrate one
TatenoPipeline per channel; odd-indexed frames are added as true negatives
and split the same way.

Output
------
For each algorithm × split:
  • Per-scene table: Accuracy / Precision / Recall / F1 / Counts
  • Aggregated confusion matrix — absolute counts
  • Aggregated confusion matrix — row-normalised percentages
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
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset, PERSON_CLASS_ID
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix, iou_bbox

try:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = "dataset"
CHANNELS = (0, 1, 2)
EMPTY_SCENE = "emptyroomwithpcscreen"
EXCLUDE_SCENES = {"2pplrun", "3pplfight"}

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# test = 1 - TRAIN_FRAC - VAL_FRAC

IOU_THRESHOLD = 0.5   # A frame is TP only if ≥1 predicted box has IoU > this with a GT box

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_indices(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Sequential 80/10/10 split of a sorted index list."""
    n = len(indices)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val   = max(0, int(n * (TRAIN_FRAC + VAL_FRAC)) - n_train)
    train = indices[:n_train]
    val   = indices[n_train:n_train + n_val]
    test  = indices[n_train + n_val:]
    return train, val, test


def _box_level(preds, gts, thr=IOU_THRESHOLD):
    """Count TP/FP/FN at the bbox level for mean-IoU reporting."""
    matched: set[int] = set()
    tp = 0
    for p in preds:
        best_j, best_iou = -1, thr
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = iou_bbox(p.bbox, g.bbox)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gts) - len(matched)


def _pred_label(preds, gts) -> int:
    """Binary prediction label with IoU criterion.

    Negative frame (GT empty): predicted positive if detector outputs anything.
    Positive frame (GT non-empty): predicted positive only if ≥1 predicted box
    achieves IoU ≥ IOU_THRESHOLD against any GT box.
    """
    if not gts:
        return 1 if preds else 0
    if not preds:
        return 0
    for p in preds:
        for g in gts:
            if iou_bbox(p.bbox, g.bbox) >= IOU_THRESHOLD:
                return 1
    return 0


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_cm(cm: BinaryConfusionMatrix, title: str) -> None:
    total = cm.total
    def pct(v): return f"{v / total * 100:5.1f}%" if total > 0 else "  N/A "

    print(f"\n  {title}")
    print(f"  {'':25s}  {'Predicted: Human':>18}  {'Predicted: Empty':>18}")
    print(f"  {'Truth: Human (Positive)':25s}  {'TP='+str(cm.tp):>8} {pct(cm.tp):>8}  {'FN='+str(cm.fn):>8} {pct(cm.fn):>8}")
    print(f"  {'Truth: Empty (Negative)':25s}  {'FP='+str(cm.fp):>8} {pct(cm.fp):>8}  {'TN='+str(cm.tn):>8} {pct(cm.tn):>8}")
    print(f"  Accuracy={cm.accuracy:.1%}  Precision={cm.precision:.1%}  "
          f"Recall={cm.recall:.1%}  F1={cm.f1:.1%}  "
          f"Correct={cm.correct}/{total}")


def _print_scene_table(results: list[dict], title: str) -> None:
    """results: list of dicts with keys scene, acc, prec, rec, f1, tp, tn, fp, fn."""
    w = 40
    print(f"\n  {title}")
    hdr = f"  {'Scene':<{w}}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'N':>5}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        print(
            f"  {r['scene']:<{w}}  "
            f"{r['acc']:>6.1%}  {r['prec']:>7.1%}  {r['rec']:>7.1%}  {r['f1']:>7.1%}  "
            f"{r['tp']:>4d}  {r['tn']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  {r['total']:>5d}"
        )
    print(sep)


def _evaluate(detector, examples: list, label: str) -> tuple[BinaryConfusionMatrix, list[dict]]:
    """Run detector on examples, return (aggregated_cm, per_scene_rows)."""
    by_scene: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for frame, gts in examples:
        preds = detector.predict(frame)
        scene = frame.metadata.get("session_id", "unknown")
        by_scene[scene][0].append(1 if gts else 0)
        by_scene[scene][1].append(_pred_label(preds, gts))

    rows: list[dict] = []
    agg = BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({
            "scene": scene,
            "acc":   cm.accuracy,
            "prec":  cm.precision,
            "rec":   cm.recall,
            "f1":    cm.f1,
            "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn,
            "total": cm.total,
        })
    return agg, rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("  HUMAN DETECTION — TRAINING & EVALUATION  (80/10/10 split, IoU > 0.5 success criterion)")
    print("=" * 72)
    print(f"\n  Dataset  : {DATASET_ROOT}")
    print(f"  Excluded : {sorted(EXCLUDE_SCENES)}")
    print(f"  Channels : {CHANNELS}")

    # -----------------------------------------------------------------------
    # 1. Load dataset index
    # -----------------------------------------------------------------------
    idx = DatasetIndex(DATASET_ROOT, sensor_profile=MLX90640)
    print(f"\n  Sessions found  : {len(idx.sessions)}")
    print(f"  Labeled sessions: {len(idx.labeled_sessions())}")

    # -----------------------------------------------------------------------
    # 2. Calibrate one TatenoPipeline per channel on even empty-room frames
    # -----------------------------------------------------------------------
    empty_session = idx.find(EMPTY_SCENE)
    pre: dict[int, TatenoPipeline] = {}
    empty_odd_by_ch: dict[int, list] = {}  # (frame_idx, Frame) for odd frames

    for ch in CHANNELS:
        calib_frames = [
            empty_session.load_frame(ch, i)
            for i in range(empty_session.n_frames)
            if i % 2 == 0
        ]
        pipe = TatenoPipeline(MLX90640)
        pipe.fit(calib_frames)
        pre[ch] = pipe
        empty_odd_by_ch[ch] = [
            (i, pipe.predict(empty_session.load_frame(ch, i)))
            for i in range(empty_session.n_frames)
            if i % 2 == 1
        ]

    print(f"\n  TatenoPipeline calibrated per channel on "
          f"{EMPTY_SCENE} (even frames).")

    # -----------------------------------------------------------------------
    # 3. Collect examples with scene-aware 80/10/10 split
    # -----------------------------------------------------------------------
    # We track examples as (preprocessed_frame, gt_detections) lists per split.
    train_ex: list = []
    val_ex:   list = []
    test_ex:  list = []

    scene_stats: list[str] = []

    # --- Labeled scenes (excluding empty room and excluded scenes) ---
    for session in idx.sessions:
        if session.scene == EMPTY_SCENE:
            continue
        if session.scene in EXCLUDE_SCENES:
            continue

        for ch in CHANNELS:
            if ch not in session.channels_with_labels:
                continue

            from thermal_algorithms.training.label_io import list_label_files, frame_index_from_label_path
            label_files = list_label_files(session.frames_dir(ch))
            labeled_idxs = sorted(frame_index_from_label_path(p) for p in label_files)
            if not labeled_idxs:
                continue

            tr_idx, va_idx, te_idx = _split_indices(labeled_idxs)

            def load_and_preprocess(frame_idx: int):
                raw = session.load_frame(ch, frame_idx)
                proc = pre[ch].predict(raw)
                from thermal_algorithms.training.label_io import load_yolo_labels
                dets = load_yolo_labels(
                    session.frames_dir(ch) / f"frame_{frame_idx:05d}.txt",
                    frame_shape=proc.shape,
                    camera_id=ch,
                    class_filter=[PERSON_CLASS_ID],
                )
                return proc, dets

            for fi in tr_idx:
                train_ex.append(load_and_preprocess(fi))
            for fi in va_idx:
                val_ex.append(load_and_preprocess(fi))
            for fi in te_idx:
                test_ex.append(load_and_preprocess(fi))

    # --- Empty room odd frames as negatives ---
    for ch in CHANNELS:
        odd_frames = empty_odd_by_ch[ch]
        idxs = [i for i, _ in odd_frames]
        frames_by_idx = {i: f for i, f in odd_frames}
        tr_idx, va_idx, te_idx = _split_indices(idxs)
        for fi in tr_idx:
            train_ex.append((frames_by_idx[fi], []))
        for fi in va_idx:
            val_ex.append((frames_by_idx[fi], []))
        for fi in te_idx:
            test_ex.append((frames_by_idx[fi], []))

    # --- Stats ---
    def _count_pos_neg(ex):
        pos = sum(1 for _, d in ex if d)
        return pos, len(ex) - pos

    tr_pos, tr_neg = _count_pos_neg(train_ex)
    va_pos, va_neg = _count_pos_neg(val_ex)
    te_pos, te_neg = _count_pos_neg(test_ex)

    print(f"\n  Split summary (frame-channel examples):")
    print(f"    Train : {len(train_ex):5d}  (pos={tr_pos:4d}, neg={tr_neg:4d})")
    print(f"    Val   : {len(val_ex):5d}  (pos={va_pos:4d}, neg={va_neg:4d})")
    print(f"    Test  : {len(test_ex):5d}  (pos={te_pos:4d}, neg={te_neg:4d})")

    # -----------------------------------------------------------------------
    # 4. Instantiate detectors
    # -----------------------------------------------------------------------
    detectors: list[tuple[str, object]] = []

    # AdaptiveThreshold — rule-based, no training
    at = AdaptiveThresholdDetector(
        MLX90640,
        c_offset=0.4,
        min_solidity=0.45,
        min_area_pixels=4,
        pixel_value_bounds=(1.0, 25.0),
    )
    at.fit([])
    detectors.append(("AdaptiveThreshold", at))

    # HOG+SVM — train on train_ex
    print("\n  Training HOGSVMDetector ...")
    hog = HOGSVMDetector(
        MLX90640,
        stride=(2, 2),
        score_threshold=0.0,
        pyramid_scales=(0.75, 1.0, 1.25, 1.5),
    )
    hog.fit(train_ex)
    print(f"    window={hog.window_size}  cell={hog.cell_size}  feat_dim={hog.feature_dim}"
          f"  scales={hog._pyramid_scales}")
    detectors.append(("HOG+SVM", hog))

    # MobileNet-SSD — train on train_ex (skipped if torch unavailable)
    if _TORCH_OK:
        print("\n  Training MobileNetSSDDetector (CUDA — RTX 2060 Super) ...")
        ssd = MobileNetSSDDetector(MLX90640, n_epochs=30, batch_size=16, learning_rate=1e-3, device="cuda")
        ssd.fit(train_ex, verbose=True)
        detectors.append(("MobileNet-SSD", ssd))
    else:
        print("\n  MobileNetSSDDetector skipped (torch not installed).")

    # -----------------------------------------------------------------------
    # 5. Evaluate each detector on val + test
    # -----------------------------------------------------------------------
    for det_name, det in detectors:
        print("\n")
        print("=" * 72)
        print(f"  {det_name}")
        print("=" * 72)

        for split_name, split_ex in [("Val (10%)", val_ex), ("Test (10%)", test_ex)]:
            print(f"\n  ── {split_name} ──")
            agg_cm, rows = _evaluate(det, split_ex, split_name)
            _print_scene_table(rows, f"Per-scene results [{split_name}]")
            _print_cm(agg_cm, f"Aggregated confusion matrix [{split_name}]")


if __name__ == "__main__":
    main()
