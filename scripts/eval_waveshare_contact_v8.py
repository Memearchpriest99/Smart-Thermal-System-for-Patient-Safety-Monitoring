"""V8 — temporal processing on top of the V7b per-frame contact stream.

V7b (blob-merge everywhere, OR X3D on crowds) plateaued at F1 44.6% deciding
each frame independently. Contacts are sustained over time, so the frame-level
errors are largely temporal noise: short false-positive flickers (people passing
close) and short false-negative dropouts (a frame where SSD splits/merges a
blob). V8 adds a temporal layer over the V7b stream and tests three smoothers:

  T1  morphology   : per-scene open (drop positive runs < L_open) + close
                     (fill negative gaps < L_close).
  T2  window-vote  : positive at t iff >= c of the centred W-frame window is
                     positive.
  T3  temporal LR  : logistic regression on per-frame cues + windowed aggregates
                     (mean pred / mean X3D / max blob-votes over +/-W), fit on
                     train, threshold tuned on val.

Frames are processed in per-scene timeline order (full session, so windows are
filled), scored on the matched val/test split. Base = V7b at its tuned params
(k=1.0, tau2=8, theta=0.9). Goal: ~60% F1.

Outputs reports/waveshare_contact_v8_results.json.
"""

from __future__ import annotations

import json
import sys
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v6 as v6
import eval_waveshare_contact_v7 as v7
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
PBASE = dict(k=1.0, tau=0, tau2=8, theta=0.9)        # V7b tuned params

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v8_results.json"
REFERENCE = {
    "Thermo-X3D":           dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V7b (per-frame base)": dict(prec=0.338, rec=0.659, f1=0.446, far=0.081),
}


# ---------------------------------------------------------------------------
# Per-frame signal extraction
# ---------------------------------------------------------------------------

def signals(rec, x3dconf):
    people, states = v7._states(rec, PBASE)
    votes = states.count("T"); merged = states.count("M")
    base = v7.v7_decision(rec, PBASE, x3dconf, True)
    cleaned = [vv.merge_oversegmented([bb for bb in c["boxes"]]) for c in rec["cams"]]
    mingap = min((vv.min_gap(b) for b in cleaned if len(b) >= 2), default=50.0)
    mingap = min(mingap, 50.0)
    strong = (votes >= 2) or (float(x3dconf[id(rec)]) > 0.9)
    return {"base": base, "votes": votes, "merged": merged,
            "x3d": float(x3dconf[id(rec)]), "people": people, "mingap": mingap,
            "strong": bool(strong)}


# ---------------------------------------------------------------------------
# Temporal operators on per-scene ordered streams
# ---------------------------------------------------------------------------

def _runs(a):
    out = []; i = 0; n = len(a)
    while i < n:
        j = i
        while j < n and a[j] == a[i]:
            j += 1
        out.append((i, j, a[i])); i = j
    return out


def morph(arr, l_open, l_close):
    a = list(arr)
    if l_open > 0:
        for s, e, v in _runs(a):
            if v == 1 and (e - s) < l_open:
                for i in range(s, e):
                    a[i] = 0
    if l_close > 0:
        for s, e, v in _runs(a):
            if v == 0 and (e - s) < l_close and s > 0 and e < len(a):
                for i in range(s, e):
                    a[i] = 1
    return a


def morph_conf(arr, strong, l_open, l_close):
    """Like morph(), but a short positive run is removed by 'open' only if it
    contains no strong-evidence frame (>=2 cameras agree or X3D very confident)."""
    a = list(arr)
    if l_open > 0:
        for s, e, v in _runs(a):
            if v == 1 and (e - s) < l_open and not any(strong[s:e]):
                for i in range(s, e):
                    a[i] = 0
    if l_close > 0:
        for s, e, v in _runs(a):
            if v == 0 and (e - s) < l_close and s > 0 and e < len(a):
                for i in range(s, e):
                    a[i] = 1
    return a


def winvote(arr, r, c):
    n = len(arr); out = [0] * n
    for i in range(n):
        lo, hi = max(0, i - r), min(n, i + r + 1)
        out[i] = 1 if sum(arr[lo:hi]) >= c else 0
    return out


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _cm_on(recs, pred_by_id, split):
    yt, yp = [], []
    for r in recs:
        if r["split"] == split:
            yt.append(r["label"]); yp.append(pred_by_id[id(r)])
    return binary_confusion_matrix(yt, yp)


def _per_scene(recs, pred_by_id):
    by = defaultdict(lambda: ([], []))
    for r in recs:
        if r["split"] == "test":
            by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(pred_by_id[id(r)])
    rows = []
    for s in sorted(by):
        a, b = by[s]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": s, "rec": c.recall, "fp": c.fp, "tp": c.tp,
                     "fn": c.fn, "total": c.total})
    return rows


def main():
    print("=" * 80)
    print("  V8 — temporal processing on the V7b stream")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD + raw/residual cache ...", flush=True)
    recs = v6.build_cache(idx, ssd, pre, labels)
    print("  [2] Thermo-X3D confidences ...", flush=True)
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)

    # Per-frame signals + per-scene ordering
    for r in recs:
        r["sig"] = signals(r, x3dconf)
    scenes = defaultdict(list)
    for r in recs:
        scenes[r["scene"]].append(r)
    for s in scenes:
        scenes[s].sort(key=lambda r: r["fi"])

    test_pos = sum(1 for r in recs if r["split"] == "test" and r["label"])
    print(f"      test pos={test_pos} / {sum(1 for r in recs if r['split']=='test')}")

    summary = []
    results = {}

    def register(name, desc, params, pred_by_id):
        cm = _cm_on(recs, pred_by_id, "test")
        summary.append((name, desc, params, cm))
        results[name] = {"desc": desc, "params": params,
                         "test": {"prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                                  "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn,
                                  "fp": cm.fp, "fn": cm.fn},
                         "per_scene": _per_scene(recs, pred_by_id)}

    # ---- Base V7b (no temporal) ----
    base_by_id = {id(r): r["sig"]["base"] for r in recs}
    register("V7b-base", "per-frame base", PBASE, base_by_id)

    # ---- T1 morphology ----
    def apply_stream(op):
        pred = {}
        for s, rs in scenes.items():
            arr = [r["sig"]["base"] for r in rs]
            sm = op(arr)
            for r, v in zip(rs, sm):
                pred[id(r)] = v
        return pred

    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            pred = apply_stream(lambda a, lo=lo, lc=lc: morph(a, lo, lc))
            cm = _cm_on(recs, pred, "val")
            if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cm, pred)
    register("T1", "morphology open+close", {"l_open": best[0][0], "l_close": best[0][1]}, best[2])

    # ---- T2 window-vote ----
    best = None
    for r_ in (1, 2, 3, 4, 5):
        W = 2 * r_ + 1
        for c in range(1, W + 1):
            pred = apply_stream(lambda a, r_=r_, c=c: winvote(a, r_, c))
            cm = _cm_on(recs, pred, "val")
            if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                best = ((r_, c), cm, pred)
    register("T2", "centred window vote", {"radius": best[0][0], "min_count": best[0][1],
                                           "window": 2 * best[0][0] + 1}, best[2])

    # ---- T4 confidence-aware morphology ----
    def apply_stream_conf(l_open, l_close):
        pred = {}
        for s, rs in scenes.items():
            arr = [r["sig"]["base"] for r in rs]
            strong = [r["sig"]["strong"] for r in rs]
            sm = morph_conf(arr, strong, l_open, l_close)
            for r, v in zip(rs, sm):
                pred[id(r)] = v
        return pred

    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            pred = apply_stream_conf(lo, lc)
            cm = _cm_on(recs, pred, "val")
            if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cm, pred)
    register("T4", "confidence-aware morphology",
             {"l_open": best[0][0], "l_close": best[0][1]}, best[2])

    # ---- T3 temporal logistic regression ----
    W = 3

    def feat(rs, i):
        g = rs[i]["sig"]
        lo, hi = max(0, i - W), min(len(rs), i + W + 1)
        win = rs[lo:hi]
        wmean_base = np.mean([w["sig"]["base"] for w in win])
        wmean_x3d = np.mean([w["sig"]["x3d"] for w in win])
        wmax_votes = max(w["sig"]["votes"] for w in win)
        wmean_people = np.mean([w["sig"]["people"] for w in win])
        return [g["base"], g["votes"], g["merged"], g["x3d"], g["people"], g["mingap"],
                wmean_base, wmean_x3d, wmax_votes, wmean_people]

    Xtr, ytr, Xva, yva, Xte, te_ids = [], [], [], [], [], []
    for s, rs in scenes.items():
        for i, r in enumerate(rs):
            f = feat(rs, i)
            if r["split"] == "train":
                Xtr.append(f); ytr.append(r["label"])
            elif r["split"] == "val":
                Xva.append(f); yva.append(r["label"])
            else:
                Xte.append(f); te_ids.append(id(r))
    Xtr, Xva, Xte = np.array(Xtr), np.array(Xva), np.array(Xte)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in (0.05, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        for thr in np.linspace(0.1, 0.9, 17):
            cm = binary_confusion_matrix(list(yva), [1 if x > thr else 0 for x in pv])
            if best is None or (cm.f1, cm.recall) > (best[2].f1, best[2].recall):
                best = (C, thr, cm, clf)
    C, thr, _, clf = best
    pte = clf.predict_proba(Xte)[:, 1]
    pred = {tid: (1 if p > thr else 0) for tid, p in zip(te_ids, pte)}
    # val/test only in pred; fill others 0 (not scored)
    for r in recs:
        pred.setdefault(id(r), 0)
    register("T3", "temporal logistic regression", {"C": C, "thr": float(thr)}, pred)

    # ---- Report ----
    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    for v, desc, p, cm in summary:
        print(f"\n  {v} — {desc}   params={p}")
        print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
              f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")

    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<30} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        print(f"  {name:<30} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}")
    for v, desc, p, cm in summary:
        if v == "V7b-base":
            continue
        print(f"  {v+' '+desc:<30} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}")
    best = max(summary, key=lambda s: s[3].f1)
    print(f"\n  Best: {best[0]} F1={best[3].f1:.1%}  (target ~60%; V7b base 44.6%).")

    # hedron + dance focus
    print("\n  Key scenes (test) — best method per-scene:")
    rows = {r["scene"]: r for r in results[best[0]]["per_scene"]}
    for s in ("2ppl_fight", "3pp_surprise", "3pplhedroncolider", "3ppl_dance", "1person cig"):
        if s in rows:
            r = rows[s]
            print(f"    {s:<20} TP={r['tp']} FN={r['fn']} FP={r['fp']} (pos={r['tp']+r['fn']})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
