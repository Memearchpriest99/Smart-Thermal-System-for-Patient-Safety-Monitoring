"""Geometric contact detector under the SAME timeline split as the deep models.

Re-scores GeometricContactDetector (min_sources=2, ε=15 px, δ swept on val) on
the identical per-session timeline split used by eval_waveshare_contact_dl.py
(train .60 / val .15 / test .25), so all three contact detectors share one
695-frame test set. Reuses the cached SSD checkpoint — no retraining.

Outputs reports/waveshare_contact_geometric_timeline.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import eval_waveshare_contact as geo
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

# Must match eval_waveshare_contact_dl.py exactly.
PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
T = 16
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
EPSILON_PX = 15.0
MIN_SOURCES = 2          # GeometricContactDetector definition (cross-camera validated)
DELTA_GRID_PX = [4, 6, 8, 10, 12, 15, 18, 22, 28, 36]

OUT_JSON = _ROOT / "reports" / "waveshare_contact_geometric_timeline.json"


def main():
    print("=" * 78)
    print("  GEOMETRIC CONTACT — timeline split (matches deep-model protocol)")
    print("=" * 78)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    print(f"  Loaded SSD checkpoint → {geo.SSD_CKPT}")
    pre = geo.fit_preprocessors(idx)
    H, _, n_track = geo.calibrate_homography(idx)
    print(f"  Homography self-calibrated ({n_track} track frames), ε={EPSILON_PX}px, min_src={MIN_SOURCES}")

    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    records = []          # (split, scene, label, actors_world)
    t0 = time.time(); n = 0
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        frame_indices = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(frame_indices) < T + 2:           # same inclusion rule as the DL harness
            continue
        n_tr = int(len(frame_indices) * TRAIN_FRAC)
        n_va = int(len(frame_indices) * (TRAIN_FRAC + VAL_FRAC))
        for k, fi in enumerate(frame_indices):
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            if split == "train":
                continue                          # geometric needs no training
            dets = tuple(ssd.predict(pre[ch].predict(geo.get_frame(session, ch, fi)))
                         for ch in CHANNELS)
            actors = geo.actors_from_dets(dets, H, EPSILON_PX, MIN_SOURCES)
            records.append((split, scene, int(labels[scene][fi]), actors))
            n += 1
            if n % 300 == 0:
                print(f"      ... {n} frames  ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)

    val = [r for r in records if r[0] == "val"]
    test = [r for r in records if r[0] == "test"]
    print(f"  val={len(val)} (pos={sum(r[2] for r in val)})  "
          f"test={len(test)} (pos={sum(r[2] for r in test)})")

    # Sweep δ on val (max F1), apply to test.
    sweep = []
    for delta in DELTA_GRID_PX:
        yt = [r[2] for r in val]
        yp = [geo._contact_pred(r[3], delta) for r in val]
        cm = binary_confusion_matrix(yt, yp)
        sweep.append((delta, cm))
        print(f"        δ={delta:>4} px  →  P={cm.precision:5.1%}  R={cm.recall:5.1%}  F1={cm.f1:5.1%}")
    best_delta, best_cm = max(sweep, key=lambda x: (x[1].f1, x[1].recall))
    print(f"\n  Best δ = {best_delta} px (val F1={best_cm.f1:.1%})")

    by_scene = defaultdict(lambda: ([], []))
    for split, scene, label, actors in test:
        by_scene[scene][0].append(label)
        by_scene[scene][1].append(geo._contact_pred(actors, best_delta))
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "far": cm.false_alarm_rate,
                     "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "total": cm.total})

    geo._print_scene_table(rows, f"Per-scene [test]  (δ={best_delta}px, ε={EPSILON_PX}px, min_src={MIN_SOURCES})")
    geo._print_cm(agg, "Aggregate [test]")

    out = {"profile": PROFILE.name, "protocol": "timeline_60_15_25", "T": T,
           "epsilon_px": EPSILON_PX, "min_sources": MIN_SOURCES, "best_delta_px": best_delta,
           "test": {"aggregate": {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall,
                                  "f1": agg.f1, "far": agg.false_alarm_rate, "tp": agg.tp,
                                  "tn": agg.tn, "fp": agg.fp, "fn": agg.fn, "total": agg.total},
                    "per_scene": rows}}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
