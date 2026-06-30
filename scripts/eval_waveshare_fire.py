"""Fire detection — evaluation on the Waveshare dataset.

Sensor : Waveshare 26984 (80×62, noise floor 0.7 °C), 3 channels.
Root   : datasets/waveshare_work.

Detectors
---------
  1. OtsuFireDetector  (rule-based, hot-pixel bypass + temporal mass gradient)
  2. FireSVMDetector   (Otsu features + RBF SVM, trained, 70/30 stratified split)

Fire-positive scenes are auto-detected from the labels: any scene containing a
class-0 (fire) bounding box. Everything else is fire-negative — crucially this
includes the multi-person scenes, which are *hard* negatives: warm human bodies
(~33 °C) the detector must NOT confuse with an ignition source.

Positive prediction := alert.level != FireLevel.SAFE.

A frame's peak temperature is reported alongside recall because the annotation
boxes the fire *object* in every frame it is visible (a cigarette, a heater),
not only frames with a hot thermal signature — so the absolute-threshold recall
ceiling is bounded by how often the boxed object is actually hot.

Outputs a readable report to stdout and JSON to
reports/waveshare_fire_results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import FireLevel
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
from thermal_algorithms.training import DatasetIndex, FireFrameDataset
from thermal_algorithms.training.metrics import (
    binary_confusion_matrix, BinaryConfusionMatrix, best_iou,
)

PROFILE = WAVESHARE_26984
DATASET_ROOT = "datasets/waveshare_work"
CHANNELS = (0, 1, 2)
TRAIN_FRAC = 0.70
OUT_JSON = Path(__file__).resolve().parent.parent / "reports" / "waveshare_fire_results.json"


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _print_scene_table(rows, title):
    w = 22
    print(f"\n  {title}")
    hdr = (f"  {'Scene':<{w}}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  "
           f"{'IoU':>6}  {'TP':>4}  {'TN':>4}  {'FP':>4}  {'FN':>4}  {'N':>5}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(sep); print(hdr); print(sep)
    for r in rows:
        iou = f"{r['iou']:>5.1%}" if r['iou'] is not None else "   - "
        print(f"  {r['scene']:<{w}}  {r['acc']:>6.1%}  {r['prec']:>7.1%}  "
              f"{r['rec']:>7.1%}  {r['f1']:>7.1%}  {iou:>6}  {r['tp']:>4d}  "
              f"{r['tn']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  {r['total']:>5d}")
    print(sep)


def _print_cm(cm: BinaryConfusionMatrix, title: str):
    total = cm.total
    pct = lambda v: f"{v/total*100:5.1f}%" if total else "  N/A "
    print(f"\n  {title}")
    print(f"  {'':25s}  {'Pred: FIRE':>18}  {'Pred: SAFE':>18}")
    print(f"  {'Truth: FIRE (Pos)':25s}  {'TP='+str(cm.tp):>8} {pct(cm.tp):>8}  {'FN='+str(cm.fn):>8} {pct(cm.fn):>8}")
    print(f"  {'Truth: SAFE (Neg)':25s}  {'FP='+str(cm.fp):>8} {pct(cm.fp):>8}  {'TN='+str(cm.tn):>8} {pct(cm.tn):>8}")
    print(f"  Accuracy={cm.accuracy:.1%}  Precision={cm.precision:.1%}  Recall={cm.recall:.1%}  "
          f"F1={cm.f1:.1%}  FalseAlarm={cm.false_alarm_rate:.1%}  Correct={cm.correct}/{total}")


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def detect_fire_scenes(ds: FireFrameDataset) -> set[str]:
    scenes = set()
    for session, examples in ds.by_session():
        if any(gt.level != FireLevel.SAFE for _, gt in examples):
            scenes.add(session.scene)
    return scenes


def eval_detector(detector, ds: FireFrameDataset):
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for session, examples in ds.by_session():
        if hasattr(detector, "reset"):
            detector.reset()
        yt, yp, ious = [], [], []
        for frame, gt in examples:
            pred = detector.predict(frame)
            gt_pos = gt.level != FireLevel.SAFE
            yt.append(1 if gt_pos else 0)
            yp.append(1 if pred.level != FireLevel.SAFE else 0)
            if gt_pos:
                gt_boxes = gt.blob_features.get("bboxes", [])
                pbox = pred.blob_features.get("bbox")
                ious.append(best_iou([pbox] if pbox else [], gt_boxes))
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": session.scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "tp": cm.tp, "tn": cm.tn,
                     "fp": cm.fp, "fn": cm.fn, "total": cm.total,
                     "iou": (sum(ious) / len(ious)) if ious else None})
    return agg, rows


def split_examples(ds: FireFrameDataset):
    """Stratified sequential split per (scene, camera, pos/neg)."""
    by_key = defaultdict(list)
    for session, examples in ds.by_session():
        for frame, gt in examples:
            pos = gt.level != FireLevel.SAFE
            by_key[(session.scene, frame.camera_id, pos)].append((frame, gt))
    train, test = [], []
    for items in by_key.values():
        n_tr = max(1, int(len(items) * TRAIN_FRAC)) if len(items) > 1 else 0
        train += items[:n_tr]
        test += items[n_tr:]
    return train, test


def order_rows(rows, fire_scenes):
    return sorted(rows, key=lambda r: (0 if r['scene'] in fire_scenes else 1, r['scene']))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("  FIRE DETECTION — Waveshare 26984 (80×62)")
    print("=" * 80)
    idx = DatasetIndex(DATASET_ROOT, sensor_profile=PROFILE)
    ds = FireFrameDataset(idx, channels=CHANNELS)
    fire_scenes = detect_fire_scenes(ds)
    n_pos = sum(1 for _, g in ds if g.level != FireLevel.SAFE)
    n_tot = len(ds)
    print(f"  Root: {DATASET_ROOT}   Sessions: {len(idx.sessions)}")
    print(f"  Fire-positive scenes: {sorted(fire_scenes)}")
    print(f"  Frame balance: {n_pos} fire / {n_tot - n_pos} safe ({n_pos/n_tot:.1%} positive)")

    results = {"profile": PROFILE.name, "dataset": DATASET_ROOT,
               "fire_scenes": sorted(fire_scenes),
               "balance": {"fire": n_pos, "safe": n_tot - n_pos, "total": n_tot},
               "detectors": {}}

    # ---------------- OtsuFireDetector ----------------
    print("\n" + "=" * 80)
    print("  OtsuFireDetector — rule-based (hot-pixel bypass, t_ign=45 t_fire=60, no training)")
    print("=" * 80)
    det = OtsuFireDetector(PROFILE)
    det.fit([])
    agg, rows = eval_detector(det, ds)
    rows = order_rows(rows, fire_scenes)
    show = [r for r in rows if r['scene'] in fire_scenes or r['fp'] > 0]
    _print_scene_table(show, "Fire scenes + any scene with a false alarm")
    n_fp_scenes = sum(1 for r in rows if r['fp'] > 0 and r['scene'] not in fire_scenes)
    n_neg_scenes = sum(1 for r in rows if r['scene'] not in fire_scenes)
    print(f"\n  Negative scenes with ≥1 false alarm: {n_fp_scenes} / {n_neg_scenes}")
    _print_cm(agg, f"Aggregate confusion (all {agg.total} frames)")
    results["detectors"]["OtsuFireDetector"] = {
        "aggregate": {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall,
                      "f1": agg.f1, "far": agg.false_alarm_rate, "tp": agg.tp, "tn": agg.tn,
                      "fp": agg.fp, "fn": agg.fn, "total": agg.total},
        "per_scene": rows, "fp_scenes": n_fp_scenes, "neg_scenes": n_neg_scenes}

    # ---------------- FireSVMDetector ----------------
    print("\n" + "=" * 80)
    print("  FireSVMDetector — RBF kernel, 70/30 stratified split")
    print("=" * 80)
    train, test = split_examples(ds)
    tr_pos = sum(1 for _, g in train if g.level != FireLevel.SAFE)
    te_pos = sum(1 for _, g in test if g.level != FireLevel.SAFE)
    print(f"  Train: {len(train)} frames (pos={tr_pos})   Test: {len(test)} frames (pos={te_pos})")

    svm = FireSVMDetector(PROFILE, kernel="rbf")
    svm.fit([f for f, _ in train], [g for _, g in train])

    by_scene = defaultdict(lambda: ([], [], []))
    for frame, gt in test:
        pred = svm.predict(frame)
        scene = frame.metadata.get("session_id", "?").split("/")[0]
        gt_pos = gt.level != FireLevel.SAFE
        by_scene[scene][0].append(1 if gt_pos else 0)
        by_scene[scene][1].append(1 if pred.level != FireLevel.SAFE else 0)
        if gt_pos:
            pbox = pred.blob_features.get("bbox")
            by_scene[scene][2].append(best_iou([pbox] if pbox else [],
                                               gt.blob_features.get("bboxes", [])))
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene, (yt, yp, ious) in by_scene.items():
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "tp": cm.tp, "tn": cm.tn,
                     "fp": cm.fp, "fn": cm.fn, "total": cm.total,
                     "iou": (sum(ious) / len(ious)) if ious else None})
    rows = order_rows(rows, fire_scenes)
    show = [r for r in rows if r['scene'] in fire_scenes or r['fp'] > 0]
    _print_scene_table(show, "Test split — fire scenes + any false-alarm scene")
    _print_cm(agg, f"Aggregate confusion (test split, {agg.total} frames)")
    results["detectors"]["FireSVMDetector"] = {
        "split": {"train": len(train), "test": len(test), "train_pos": tr_pos, "test_pos": te_pos},
        "aggregate": {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall,
                      "f1": agg.f1, "far": agg.false_alarm_rate, "tp": agg.tp, "tn": agg.tn,
                      "fp": agg.fp, "fn": agg.fn, "total": agg.total},
        "per_scene": rows}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
