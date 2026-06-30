"""V4 — compound image-plane touch detector.

Stacks the three improvements that proved worthwhile in the V3.x study:

  * V3.4 box-merge   : merge over-segmented boxes (one person split in two)
                       before voting  -> cleaner per-camera people estimate.
  * V3.1 merge-aware : a single merged blob counts toward the quorum (>=1 real
                       TOUCH + TOUCH+MERGED>=2, no CLEAR) -> recovers the
                       blob-merge fights (recall engine).
  * X3D crowd-veto   : the merge-aware rule is precision-poor on CROWDS (3+
                       people whose boxes abut without contact). A blanket
                       Thermo-X3D veto can't be used because X3D itself misses
                       the 2-person fights -> instead the veto is GATED on crowd:
                          final = V3raw AND (people<=2 OR x3d_conf > theta)
                       i.e. trust the box logic on pairs (where V3 wins / X3D
                       fails), defer to X3D only when 3+ people are present
                       (where V3 sprays false alarms / X3D is reliable).

"people" = max over cameras of the merged-box count. Thresholds (tau, tau2,
theta) tuned on val, reported on the matched 695-frame test set. Goal: beat
Thermo-X3D's standalone F1 of 42.3%.

Outputs reports/waveshare_contact_v4_results.json.
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

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
TAU_GRID = [0, 1, 2, 3, 4, 6, 8, 10]
TAU2_EXTRA = [0, 2, 4, 8]
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CROWD_PEOPLE = 3                      # >=3 people -> defer to X3D

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v4_results.json"

# Reference test numbers (matched 695-frame set).
REFERENCE = {
    "Thermo-X3D (best so far)": dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V3.0 baseline":            dict(prec=0.192, rec=0.561, f1=0.286, far=0.148),
    "V3.4 box-merge":           dict(prec=0.220, rec=0.585, f1=0.320, far=0.130),
    "V3.1 merge-aware":         dict(prec=0.167, rec=0.927, f1=0.283, far=0.291),
}


def v4_decision(r, params, x3dconf):
    t, t2, theta = params
    cleaned = [vv.merge_oversegmented(b) for b in r["boxes"]]
    states = [vv._state(len(b), vv.min_gap(b), t, t2) for b in cleaned]
    votes = states.count("T"); merged = states.count("M"); clear = "C" in states
    raw = (votes >= 1 and votes + merged >= 2 and not clear)   # V3.4+V3.1
    if not raw:
        return 0
    people = max(len(b) for b in cleaned)
    if people >= CROWD_PEOPLE:                                 # crowd -> X3D arbitrates
        return 1 if x3dconf[id(r)] > theta else 0
    return 1                                                   # pair -> trust boxes


def _eval(recs, params, x3dconf):
    yt = [r["label"] for r in recs]
    yp = [v4_decision(r, params, x3dconf) for r in recs]
    return binary_confusion_matrix(yt, yp), yp


def main():
    print("=" * 80)
    print("  V4 — compound: box-merge + merge-aware + X3D crowd-veto")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD boxes over val+test ...", flush=True)
    t0 = time.time()
    recs = vv.build_boxes(idx, ssd, pre, labels)
    val = [r for r in recs if r["split"] == "val"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"      val={len(val)} (pos={sum(r['label'] for r in val)})  "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})  ({time.time()-t0:.0f}s)")

    print("  [2] Thermo-X3D confidences ...", flush=True)
    t0 = time.time()
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)
    print(f"      {len(x3dconf)} frames scored  ({time.time()-t0:.0f}s)")

    # ---- Tune (tau, tau2, theta) on val ----
    yt_val = [r["label"] for r in val]
    best = None
    for t in TAU_GRID:
        for e in TAU2_EXTRA:
            for th in THETA_GRID:
                params = (t, t + e, th)
                cm, _ = _eval(val, params, x3dconf)
                if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                    best = (params, cm)
    params = best[0]
    print(f"\n  Tuned params: tau={params[0]}px, tau2={params[1]}px, "
          f"X3D theta={params[2]}  (val F1={best[1].f1:.1%})")

    # ---- Test ----
    cm, yp = _eval(test, params, x3dconf)
    print("\n" + "=" * 80)
    print("  V4 — confusion matrix (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    geo._print_cm(cm, "V4: box-merge + merge-aware + X3D crowd-veto")

    # per scene
    by = defaultdict(lambda: ([], []))
    for r, p in zip(test, yp):
        by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(p)
    rows = []
    for scene in sorted(by):
        a, b = by[scene]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": scene, "acc": c.accuracy, "prec": c.precision, "rec": c.recall,
                     "f1": c.f1, "far": c.false_alarm_rate, "tp": c.tp, "tn": c.tn,
                     "fp": c.fp, "fn": c.fn, "total": c.total})
    geo._print_scene_table(rows, "Per-scene [test] — V4")

    # ---- Comparison ----
    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<26} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        star = "  <-- to beat" if name.startswith("Thermo") else ""
        print(f"  {name:<26} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}{star}")
    print(f"  {'V4 compound':<26} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
          f"{cm.false_alarm_rate:>7.1%}  <== NEW")
    verdict = "BEATS" if cm.f1 > REFERENCE["Thermo-X3D (best so far)"]["f1"] else "does NOT beat"
    print(f"\n  V4 {verdict} Thermo-X3D on F1 "
          f"({cm.f1:.1%} vs {REFERENCE['Thermo-X3D (best so far)']['f1']:.1%}).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"params": {"tau": params[0], "tau2": params[1], "x3d_theta": params[2],
                    "crowd_people": CROWD_PEOPLE},
         "test": {"acc": cm.accuracy, "prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                  "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn},
         "per_scene": rows}, indent=2))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
