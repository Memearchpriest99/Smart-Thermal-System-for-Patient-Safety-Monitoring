"""V7 — blob-merge applied to ALL regimes (no X3D crowd routing).

V6.3 (blob-count merge) won at F1 44.6% but deferred 3+ person CROWDS to
Thermo-X3D, which fails 3pplhedroncolider (a 3-person collision: 0/7 recall,
16 FP). Hypothesis: the connected-blob merge test does not spray on crowds the
way the gap test did (dance = separate blobs), so we can drop the X3D crowd
routing entirely and run blob-merge uniformly — which should recover the
collision scene.

  V7   : people<=1 -> 0 ; otherwise blob-merge merge-aware quorum (NO X3D).
  V7b  : same, but on crowds (people>=3) also fire if Thermo-X3D conf > theta
         (blob-merge OR X3D), to keep X3D's crowd wins as a safety net.

Per-camera TOUCH = the two nearest person boxes lie in the same connected warm
component (reuses eval_waveshare_contact_v6 helpers). Tuned on val, reported on
the matched 695-frame test set.

Outputs reports/waveshare_contact_v7_results.json.
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
import eval_waveshare_contact_v6 as v6
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v7_results.json"
REFERENCE = {
    "Thermo-X3D":          dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V6.3 blob (best box)": dict(prec=0.352, rec=0.610, f1=0.446, far=0.070),
}


def _states(rec, p):
    cleaned = [vv.merge_oversegmented([bb for bb in c["boxes"]]) for c in rec["cams"]]
    people = max((len(b) for b in cleaned), default=0)
    states = [v6._cam_state(cleaned[c], rec["cams"][c]["resid"], rec["cams"][c]["raw"], "blob", p)
              for c in range(3)]
    return people, states


def _quorum(states):
    votes = states.count("T"); merged = states.count("M"); clear = "C" in states
    return votes >= 1 and votes + merged >= 2 and not clear


def v7_decision(rec, p, x3dconf, use_x3d_on_crowd):
    people, states = _states(rec, p)
    if people <= 1:
        return 0
    blob = _quorum(states)
    if use_x3d_on_crowd and people >= 3:
        return 1 if (blob or x3dconf[id(rec)] > p["theta"]) else 0
    return 1 if blob else 0


def _tune(val, test, grid, decide_for):
    yt = [r["label"] for r in val]
    best = None
    for p in grid:
        yp = [decide_for(p)(r) for r in val]
        cm = binary_confusion_matrix(yt, yp)
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (p, cm)
    yp = [decide_for(best[0])(r) for r in test]
    cm = binary_confusion_matrix([r["label"] for r in test], yp)
    return best[0], cm, yp


def _per_scene(test, yp):
    by = defaultdict(lambda: ([], []))
    for r, p in zip(test, yp):
        by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(p)
    rows = []
    for s in sorted(by):
        a, b = by[s]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": s, "rec": c.recall, "fp": c.fp, "tp": c.tp,
                     "fn": c.fn, "total": c.total})
    return rows


def main():
    print("=" * 80)
    print("  V7 — blob-merge on ALL regimes (drop X3D crowd routing)")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD + raw/residual cache ...", flush=True)
    recs = v6.build_cache(idx, ssd, pre, labels)
    val = [r for r in recs if r["split"] == "val"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"      val={len(val)} (pos={sum(r['label'] for r in val)})  "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})")

    print("  [2] Thermo-X3D confidences (for V7b) ...", flush=True)
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)

    grid = [dict(k=k, tau=0, tau2=t2) for k in (0.5, 1.0, 1.5) for t2 in (4, 8, 12)]
    grid_b = [dict(k=k, tau=0, tau2=t2, theta=th)
              for k in (0.5, 1.0, 1.5) for t2 in (4, 8, 12) for th in (0.8, 0.9)]

    summary = []
    p7, cm7, yp7 = _tune(val, test, grid,
                         lambda p: (lambda r: v7_decision(r, p, x3dconf, False)))
    summary.append(("V7", "blob-merge everywhere", p7, cm7, yp7))

    p7b, cm7b, yp7b = _tune(val, test, grid_b,
                            lambda p: (lambda r: v7_decision(r, p, x3dconf, True)))
    summary.append(("V7b", "blob OR X3D on crowds", p7b, cm7b, yp7b))

    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    results = {}
    for v, desc, p, cm, yp in summary:
        print(f"\n  {v} — {desc}   params={p}")
        print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
              f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")
        results[v] = {"desc": desc, "params": p,
                      "test": {"acc": cm.accuracy, "prec": cm.precision, "rec": cm.recall,
                               "f1": cm.f1, "far": cm.false_alarm_rate, "tp": cm.tp,
                               "tn": cm.tn, "fp": cm.fp, "fn": cm.fn},
                      "per_scene": _per_scene(test, yp)}

    # hedron focus
    print("\n  3pplhedroncolider (the scene V6.3/X3D failed):")
    for v, desc, p, cm, yp in summary:
        row = next(r for r in results[v]["per_scene"] if r["scene"] == "3pplhedroncolider")
        print(f"    {v}: TP={row['tp']} FN={row['fn']} FP={row['fp']}  (pos={row['tp']+row['fn']})")

    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<26} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        print(f"  {name:<26} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}")
    for v, desc, p, cm, yp in summary:
        print(f"  {v+' '+desc:<26} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}")
    best = max(summary, key=lambda s: s[3].f1)
    print(f"\n  Best V7: {best[0]} F1={best[3].f1:.1%}  "
          f"(V6.3 was 44.6%, Thermo-X3D 41.7%).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
