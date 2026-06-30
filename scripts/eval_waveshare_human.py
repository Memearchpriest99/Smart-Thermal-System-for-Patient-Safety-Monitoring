"""Human detection — training & evaluation on the Waveshare dataset.

Sensor : Waveshare 26984 (80×62, noise floor 0.7 °C), 3 channels.
Root   : datasets/waveshare_work  (layout B — scene/ch{N}_*).

Detectors
---------
  1. AdaptiveThreshold  (rule-based, no training)
  2. HOG + Linear SVM   (classical ML, trained)
  3. MobileNet-SSD      (deep, PyTorch/CUDA, trained — skipped if torch missing)

Protocol
--------
* `empty_room` is the only truly empty scene (0 person boxes). Even-indexed
  empty frames calibrate one TatenoPipeline per channel; odd-indexed empty
  frames become true negatives. No leak between calibration and evaluation.
* Every other labeled scene is a positive scene (person present). Person
  bboxes come from YOLO class 1.
* Within each (scene, channel), labeled frame indices are sorted and split
  sequentially 80/10/10 → train/val/test (no temporal leakage).
* All frames are preprocessed with their channel's Tateno background residual
  before detection (matches the runtime pipeline).
* A frame is a positive prediction iff ≥1 predicted box reaches IoU ≥ 0.5 with
  a ground-truth box (empty frames: positive iff any box is emitted).

Outputs a readable report to stdout and a machine-readable JSON to
reports/waveshare_human_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, PERSON_CLASS_ID
from thermal_algorithms.training.label_io import (
    list_label_files,
    frame_index_from_label_path,
    load_yolo_labels,
)
from thermal_algorithms.training.metrics import (
    binary_confusion_matrix,
    BinaryConfusionMatrix,
    iou_bbox,
)

try:
    import torch
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    _TORCH_OK = False
    _DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILE = WAVESHARE_26984
DATASET_ROOT = "datasets/waveshare_work"
CHANNELS = (0, 1, 2)
EMPTY_SCENE = "empty_room"

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
IOU_THRESHOLD = 0.5

OUT_JSON = Path(__file__).resolve().parent.parent / "reports" / "waveshare_human_results.json"


# ---------------------------------------------------------------------------
# Split + evaluation helpers
# ---------------------------------------------------------------------------

def _split_indices(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
    n = len(indices)
    n_train = max(1, int(n * TRAIN_FRAC))
    n_val = max(0, int(n * (TRAIN_FRAC + VAL_FRAC)) - n_train)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def _pred_label(preds, gts) -> int:
    """Binary presence label using the IoU success criterion."""
    if not gts:
        return 1 if preds else 0
    if not preds:
        return 0
    for p in preds:
        for g in gts:
            if iou_bbox(p.bbox, g.bbox) >= IOU_THRESHOLD:
                return 1
    return 0


def _evaluate(detector, examples):
    """Single predict pass per frame → (aggregate CM, per-scene rows, mean IoU).

    The detector is run exactly once per frame; both the binary presence label
    (Criterion B, IoU>=0.5) and the per-GT-box best IoU are derived from that
    single prediction. This avoids the pathological double-pass that made HOG
    evaluation take hours.
    """
    by_scene: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    iou_scores: list[float] = []
    t0 = time.time()
    for n, (frame, gts) in enumerate(examples):
        preds = detector.predict(frame)
        scene = frame.metadata.get("session_id", "unknown")
        by_scene[scene][0].append(1 if gts else 0)
        by_scene[scene][1].append(_pred_label(preds, gts))
        for g in gts:
            iou_scores.append(max((iou_bbox(p.bbox, g.bbox) for p in preds), default=0.0))
        if (n + 1) % 200 == 0:
            print(f"      ... {n+1}/{len(examples)} frames  ({(time.time()-t0)/(n+1)*1000:.0f} ms/frame)",
                  flush=True)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "far": cm.false_alarm_rate,
                     "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "total": cm.total})
    return agg, rows, mean_iou


def _print_scene_table(rows, title):
    w = 24
    print(f"\n  {title}")
    hdr = (f"  {'Scene':<{w}}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
           f"{'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'N':>5}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep); print(hdr); print(sep)
    for r in rows:
        print(f"  {r['scene']:<{w}}  {r['acc']:>6.1%}  {r['prec']:>7.1%}  "
              f"{r['rec']:>7.1%}  {r['f1']:>7.1%}  {r['tp']:>4d}  {r['tn']:>4d}  "
              f"{r['fp']:>4d}  {r['fn']:>4d}  {r['total']:>5d}")
    print(sep)


def _print_cm(cm: BinaryConfusionMatrix, title: str):
    total = cm.total
    pct = lambda v: f"{v/total*100:5.1f}%" if total else "  N/A "
    print(f"\n  {title}")
    print(f"  {'':25s}  {'Pred: Human':>18}  {'Pred: Empty':>18}")
    print(f"  {'Truth: Human (Pos)':25s}  {'TP='+str(cm.tp):>8} {pct(cm.tp):>8}  {'FN='+str(cm.fn):>8} {pct(cm.fn):>8}")
    print(f"  {'Truth: Empty (Neg)':25s}  {'FP='+str(cm.fp):>8} {pct(cm.fp):>8}  {'TN='+str(cm.tn):>8} {pct(cm.tn):>8}")
    print(f"  Accuracy={cm.accuracy:.1%}  Precision={cm.precision:.1%}  Recall={cm.recall:.1%}  "
          f"F1={cm.f1:.1%}  FalseAlarm={cm.false_alarm_rate:.1%}  Correct={cm.correct}/{total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print("  HUMAN DETECTION — Waveshare 26984 (80×62)  —  80/10/10 split, IoU≥0.5")
    print("=" * 78)
    idx = DatasetIndex(DATASET_ROOT, sensor_profile=PROFILE)
    print(f"  Root: {DATASET_ROOT}   Sessions: {len(idx.sessions)}   "
          f"Labeled: {len(idx.labeled_sessions())}")
    print(f"  Torch: {_TORCH_OK}  Device: {_DEVICE}")

    empty = idx.find(EMPTY_SCENE)

    # 1. Per-channel Tateno calibration on even empty frames; odd → negatives.
    pre: dict[int, TatenoPipeline] = {}
    empty_neg_by_ch: dict[int, list[tuple[int, object]]] = {}
    for ch in CHANNELS:
        calib = [empty.load_frame(ch, i) for i in range(empty.n_frames) if i % 2 == 0]
        pre[ch] = TatenoPipeline(PROFILE).fit(calib)
        empty_neg_by_ch[ch] = [
            (i, pre[ch].predict(empty.load_frame(ch, i)))
            for i in range(empty.n_frames) if i % 2 == 1
        ]
    print(f"  TatenoPipeline calibrated per channel on '{EMPTY_SCENE}' (even frames).")

    # 2. Build examples with 80/10/10 split.
    train_ex, val_ex, test_ex = [], [], []

    for session in idx.sessions:
        if session.scene == EMPTY_SCENE:
            continue
        for ch in CHANNELS:
            if ch not in session.channels_with_labels:
                continue
            label_files = list_label_files(session.frames_dir(ch))
            labeled = sorted(frame_index_from_label_path(p) for p in label_files)
            if not labeled:
                continue
            tr, va, te = _split_indices(labeled)

            def load(fi: int):
                proc = pre[ch].predict(session.load_frame(ch, fi))
                dets = load_yolo_labels(
                    session.frames_dir(ch) / f"frame_{fi:05d}.txt",
                    frame_shape=proc.shape, camera_id=ch, class_filter=[PERSON_CLASS_ID])
                return proc, dets

            train_ex += [load(fi) for fi in tr]
            val_ex += [load(fi) for fi in va]
            test_ex += [load(fi) for fi in te]

    # Empty-room odd frames as negatives, same split.
    for ch in CHANNELS:
        idxs = [i for i, _ in empty_neg_by_ch[ch]]
        fmap = dict(empty_neg_by_ch[ch])
        tr, va, te = _split_indices(idxs)
        train_ex += [(fmap[i], []) for i in tr]
        val_ex += [(fmap[i], []) for i in va]
        test_ex += [(fmap[i], []) for i in te]

    cnt = lambda ex: (sum(1 for _, d in ex if d), sum(1 for _, d in ex if not d))
    tr_p, tr_n = cnt(train_ex); va_p, va_n = cnt(val_ex); te_p, te_n = cnt(test_ex)
    print(f"\n  Split (frame-channel examples):")
    print(f"    Train : {len(train_ex):5d}  (pos={tr_p:4d}, neg={tr_n:4d})")
    print(f"    Val   : {len(val_ex):5d}  (pos={va_p:4d}, neg={va_n:4d})")
    print(f"    Test  : {len(test_ex):5d}  (pos={te_p:4d}, neg={te_n:4d})")

    # 3. Detectors.
    detectors = []

    at = AdaptiveThresholdDetector(PROFILE, c_offset=0.4, min_solidity=0.45,
                                   min_area_pixels=8, pixel_value_bounds=(1.0, 25.0)).fit([])
    detectors.append(("AdaptiveThreshold", at))

    # HOG predict is sliding-window over an image pyramid; at 80x62 (6.5x the
    # MLX pixel count) a dense stride/4-scale config costs ~5 s/frame. stride
    # (4,4) with two scales keeps recall while making evaluation tractable.
    print("\n  Training HOG+SVM ...", flush=True)
    t = time.time()
    hog = HOGSVMDetector(PROFILE, stride=(4, 4), score_threshold=0.0,
                         pyramid_scales=(1.0, 1.25))
    hog.fit(train_ex)
    print(f"    window={hog.window_size} cell={hog.cell_size} feat_dim={hog.feature_dim} "
          f"({time.time()-t:.1f}s)", flush=True)
    detectors.append(("HOG+SVM", hog))

    if _TORCH_OK:
        print(f"\n  Training MobileNet-SSD ({_DEVICE}) ...", flush=True)
        t = time.time()
        ssd = MobileNetSSDDetector(PROFILE, n_epochs=30, batch_size=16,
                                   learning_rate=1e-3, device=_DEVICE)
        ssd.fit(train_ex, verbose=True)
        print(f"    MobileNet-SSD trained ({time.time()-t:.1f}s)", flush=True)
        detectors.append(("MobileNet-SSD", ssd))
    else:
        print("\n  MobileNet-SSD skipped (torch unavailable).")

    # 4. Evaluate.
    results = {"profile": PROFILE.name, "dataset": DATASET_ROOT, "device": _DEVICE,
               "split": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex),
                         "test_pos": te_p, "test_neg": te_n},
               "detectors": {}}

    for name, det in detectors:
        print("\n" + "=" * 78)
        print(f"  {name}")
        print("=" * 78)
        det_out = {}
        for split_name, ex in [("val", val_ex), ("test", test_ex)]:
            print(f"\n  Evaluating {name} on {split_name} ({len(ex)} frames) ...", flush=True)
            agg, rows, miou = _evaluate(det, ex)
            _print_scene_table(rows, f"Per-scene [{split_name}]")
            _print_cm(agg, f"Aggregate [{split_name}]  (mean IoU on positives = {miou:.1%})")
            det_out[split_name] = {
                "aggregate": {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall,
                              "f1": agg.f1, "far": agg.false_alarm_rate, "mean_iou": miou,
                              "tp": agg.tp, "tn": agg.tn, "fp": agg.fp, "fn": agg.fn,
                              "total": agg.total},
                "per_scene": rows,
            }
        results["detectors"][name] = det_out

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
