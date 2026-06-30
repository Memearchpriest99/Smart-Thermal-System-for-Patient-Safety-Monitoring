"""V5 — people-count router (box logic on pairs, Thermo-X3D on crowds).

V4 showed the two methods are complementary: the box logic catches the 2-person
fights that Thermo-X3D misses, while X3D stays quiet on the 3-person crowds the
box logic floods with false alarms. V4 combined them with an AND on crowds;
V5 instead *routes* each frame to whichever detector owns that regime:

    people = max over cameras of merged-box count   (distinct subjects seen)
      people <= 1  -> 0           (one subject -> contact impossible)
      people == 2  -> box logic   (merge-aware quorum; X3D is weak on fights)
      people >= 3  -> Thermo-X3D  (conf > theta; box logic sprays on crowds)

V4's residual false alarms were dominated by single-person scenes where SSD
detects a hot *object* (cigarette tip, heater) as a second "person". V5 adds a
minimum box-area filter (tuned) to drop those tiny spurious detections before
routing.

Thresholds (a_min, tau, tau2, theta) tuned on val, reported on the matched
695-frame test set. Goal: beat Thermo-X3D's F1 of 42.3%.

Outputs reports/waveshare_contact_v5_results.json.
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

AMIN_GRID = [0, 15, 30, 50]          # min box area (px^2) to drop object/noise boxes
TAU_GRID = [0, 1, 2, 3, 4, 6, 8, 10]
TAU2_EXTRA = [0, 2, 4, 8]
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v5_results.json"

REFERENCE = {
    "Thermo-X3D (to beat)": dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V3.0 baseline":        dict(prec=0.192, rec=0.561, f1=0.286, far=0.148),
    "V4 compound":          dict(prec=0.243, rec=0.659, f1=0.355, far=0.128),
}


def _filter_area(boxes, a_min):
    if a_min <= 0:
        return boxes
    return [b for b in boxes if b[2] * b[3] >= a_min]


def v5_decision(r, params, x3dconf):
    a_min, t, t2, theta = params
    cleaned = [vv.merge_oversegmented(_filter_area(b, a_min)) for b in r["boxes"]]
    people = max((len(b) for b in cleaned), default=0)
    if people <= 1:
        return 0
    if people == 2:
        states = [vv._state(len(b), vv.min_gap(b), t, t2) for b in cleaned]
        votes = states.count("T"); merged = states.count("M"); clear = "C" in states
        return 1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0
    return 1 if x3dconf[id(r)] > theta else 0          # crowd -> X3D


def _eval(recs, params, x3dconf):
    yt = [r["label"] for r in recs]
    yp = [v5_decision(r, params, x3dconf) for r in recs]
    return binary_confusion_matrix(yt, yp), yp


def main():
    print("=" * 80)
    print("  V5 — people-count router (pairs->boxes, crowds->Thermo-X3D)")
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

    # ---- Tune on val ----
    yt_val = [r["label"] for r in val]
    best = None
    for a_min in AMIN_GRID:
        for t in TAU_GRID:
            for e in TAU2_EXTRA:
                for th in THETA_GRID:
                    params = (a_min, t, t + e, th)
                    cm, _ = _eval(val, params, x3dconf)
                    if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                        best = (params, cm)
    params = best[0]
    print(f"\n  Tuned: a_min={params[0]}px^2  tau={params[1]}px  tau2={params[2]}px  "
          f"X3D theta={params[3]}  (val F1={best[1].f1:.1%})")

    # ---- Test ----
    cm, yp = _eval(test, params, x3dconf)
    print("\n" + "=" * 80)
    print("  V5 — confusion matrix (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    geo._print_cm(cm, "V5: people-count router")

    by = defaultdict(lambda: ([], []))
    for r, p in zip(test, yp):
        by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(p)
    rows = []
    for scene in sorted(by):
        a, b = by[scene]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": scene, "acc": c.accuracy, "prec": c.precision, "rec": c.recall,
                     "f1": c.f1, "far": c.false_alarm_rate, "tp": c.tp, "tn": c.tn,
                     "fp": c.fp, "fn": c.fn, "total": c.total})
    geo._print_scene_table(rows, "Per-scene [test] — V5")

    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<24} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        star = "  <-- to beat" if "to beat" in name else ""
        print(f"  {name:<24} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}{star}")
    print(f"  {'V5 router':<24} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
          f"{cm.false_alarm_rate:>7.1%}  <== NEW")
    x3d_f1 = REFERENCE["Thermo-X3D (to beat)"]["f1"]
    print(f"\n  V5 {'BEATS' if cm.f1 > x3d_f1 else 'does NOT beat'} Thermo-X3D on F1 "
          f"({cm.f1:.1%} vs {x3d_f1:.1%}).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"params": {"a_min": params[0], "tau": params[1], "tau2": params[2], "x3d_theta": params[3]},
         "test": {"acc": cm.accuracy, "prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                  "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn},
         "per_scene": rows}, indent=2))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
