"""V9 — two-body-merge spatial test on crowds + T1 temporal morphology.

V8/T1 reached F1 53.8%; the entire remaining gap to 60% is 3pplhedroncolider's
19 SUSTAINED false positives — a spatial problem (in a dense 3-body cluster,
"a pair is in contact" and "the whole crowd is one blob" look identical).

V9 adds a spatial discriminator on crowds. Instead of firing whenever any two
people share a connected warm component, it requires a *two-body* merge:

  per camera (people >= 3):
     TOUCH iff some connected warm component contains EXACTLY 2 person-box
     centres (a pair fused, the rest separate).  An all-3-in-one-blob cluster
     -> no fire (that's the ambiguous collision).
  per camera (people == 2):
     TOUCH iff the two share a component (unchanged).

Decision: people<=1 -> 0 ; merge-aware quorum across cameras ; OR Thermo-X3D on
crowds. Then T1 morphology (open+close) on the per-scene stream, tuned on val.
Base blob params fixed at V7b's (k=1.0, tau2=8, theta=0.9); morphology tuned.

Outputs reports/waveshare_contact_v9_results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from scipy.ndimage import label as cc_label

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v6 as v6
import eval_waveshare_contact_v8 as v8
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
H_PX, W_PX = PROFILE.height, PROFILE.width
K, TAU2, THETA = 1.0, 8.0, 0.9          # fixed base params (from V7b)

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v9_results.json"
REFERENCE = {
    "Thermo-X3D":        dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V7b per-frame":     dict(prec=0.338, rec=0.659, f1=0.446, far=0.081),
    "V8/T1 (temporal)":  dict(prec=0.444, rec=0.683, f1=0.538, far=0.054),
}


def _comp_counts(resid, boxes, k):
    thr = float(resid.mean() + k * resid.std())
    lab, _ = cc_label(resid > thr)
    cnt = Counter()
    for b in boxes:
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        l = lab[int(np.clip(cy, 0, H_PX - 1)), int(np.clip(cx, 0, W_PX - 1))]
        if l > 0:
            cnt[l] += 1
    return cnt


def _cam_state_2body(boxes, resid, k, tau2):
    n = len(boxes)
    if n == 0:
        return "C"
    if n == 1:
        return "M"
    cnt = _comp_counts(resid, boxes, k)
    any_merge = any(v >= 2 for v in cnt.values())
    pair_merge = any(v == 2 for v in cnt.values())
    touch = any_merge if n == 2 else pair_merge        # crowds need an exactly-2 merge
    if touch:
        return "T"
    return "N" if vv.min_gap(boxes) < tau2 else "C"


def v9_eval(rec, x3dconf):
    """Returns (decision, strong) where strong = high-confidence frame
    (>=2 cameras independently show a two-body merge, or X3D very confident)."""
    cleaned = [vv.merge_oversegmented([bb for bb in c["boxes"]]) for c in rec["cams"]]
    people = max((len(b) for b in cleaned), default=0)
    if people <= 1:
        return 0, False
    states = [_cam_state_2body(cleaned[c], rec["cams"][c]["resid"], K, TAU2) for c in range(3)]
    votes = states.count("T"); merged = states.count("M"); clear = "C" in states
    blob = votes >= 1 and votes + merged >= 2 and not clear
    if people >= 3:
        dec = 1 if (blob or x3dconf[id(rec)] > THETA) else 0
    else:
        dec = 1 if blob else 0
    strong = (votes >= 2) or (x3dconf[id(rec)] > 0.9)
    return dec, strong


def v9_decision(rec, x3dconf):
    return v9_eval(rec, x3dconf)[0]


def main():
    print("=" * 80)
    print("  V9 — two-body-merge crowd test + temporal morphology")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD + raw/residual cache ...", flush=True)
    recs = v6.build_cache(idx, ssd, pre, labels)
    print("  [2] Thermo-X3D confidences ...", flush=True)
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)

    for r in recs:
        r["v9"], r["strong"] = v9_eval(r, x3dconf)
    scenes = defaultdict(list)
    for r in recs:
        scenes[r["scene"]].append(r)
    for s in scenes:
        scenes[s].sort(key=lambda r: r["fi"])

    def cm_split(pred_by_id, split):
        yt = [r["label"] for r in recs if r["split"] == split]
        yp = [pred_by_id[id(r)] for r in recs if r["split"] == split]
        return binary_confusion_matrix(yt, yp)

    def per_scene(pred_by_id):
        by = defaultdict(lambda: ([], []))
        for r in recs:
            if r["split"] == "test":
                by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(pred_by_id[id(r)])
        rows = []
        for s in sorted(by):
            a, b = by[s]; c = binary_confusion_matrix(a, b)
            rows.append({"scene": s, "tp": c.tp, "fn": c.fn, "fp": c.fp})
        return rows

    summary = []
    results = {}

    # V9 spatial only (no temporal)
    raw_pred = {id(r): r["v9"] for r in recs}
    cm = cm_split(raw_pred, "test")
    summary.append(("V9 (spatial only)", cm)); results["V9"] = {"per_scene": per_scene(raw_pred)}

    # V9 + T1 morphology (tune on val)
    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            pred = {}
            for s, rs in scenes.items():
                arr = [r["v9"] for r in rs]
                sm = v8.morph(arr, lo, lc)
                for r, v in zip(rs, sm):
                    pred[id(r)] = v
            cmv = cm_split(pred, "val")
            if best is None or (cmv.f1, cmv.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cmv, pred)
    cm_t = cm_split(best[2], "test")
    summary.append((f"V9 + T1 morph (lo={best[0][0]},lc={best[0][1]})", cm_t))
    results["V9_T1"] = {"params": {"l_open": best[0][0], "l_close": best[0][1]},
                        "per_scene": per_scene(best[2])}

    # V9 + confidence-aware morphology (preserve strong short runs)
    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            pred = {}
            for s, rs in scenes.items():
                arr = [r["v9"] for r in rs]; strong = [r["strong"] for r in rs]
                sm = v8.morph_conf(arr, strong, lo, lc)
                for r, v in zip(rs, sm):
                    pred[id(r)] = v
            cmv = cm_split(pred, "val")
            if best is None or (cmv.f1, cmv.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cmv, pred)
    cm_c = cm_split(best[2], "test")
    summary.append((f"V9 + conf-morph (lo={best[0][0]},lc={best[0][1]})", cm_c))
    results["V9_T4"] = {"params": {"l_open": best[0][0], "l_close": best[0][1]},
                        "per_scene": per_scene(best[2])}

    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    for name, cm in summary:
        print(f"\n  {name}")
        print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
              f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")
        results.setdefault(name, {})
        results[name]["test"] = {"prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                                 "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn,
                                 "fp": cm.fp, "fn": cm.fn}

    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<28} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for nm, m in REFERENCE.items():
        print(f"  {nm:<28} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}")
    for nm, cm in summary:
        print(f"  {nm:<28} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}")

    print("\n  3pplhedroncolider (the target scene), V9+T1:")
    row = next(r for r in results["V9_T1"]["per_scene"] if r["scene"] == "3pplhedroncolider")
    print(f"    TP={row['tp']} FN={row['fn']} FP={row['fp']}  (was FP=19 in V8/T1)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
