"""V6 — thermal-aware optimizations of the homography-free family.

The box family plateaued (V4/V5 F1 35.5%) because it ignores the thermal data:
its errors are thermal questions (a hot OBJECT misread as a person; image
proximity vs. real CONTACT). V6 puts the temperature signal back in. Each
variant builds on the V4 people-count router (pairs -> box logic, crowds ->
Thermo-X3D) and adds one thermal cue, tuned on val, reported on the matched
695-frame test set.

  V6.1  heat-bridge      : a camera votes TOUCH only if the boxes are near AND
                           the residual along the line between them stays warm
                           (a continuous body bridge), not a cool gap.
  V6.2  temp box reject  : drop SSD boxes whose peak temperature is fire-like
                           (cigarette/heater), so a hot object is no longer a
                           second "person". This also corrects the people count.
  V6.3  blob-count       : a camera votes TOUCH if the two nearest person boxes
                           fall in the SAME connected warm component (bodies
                           fused), via connected-component labelling.
  V6.4  learned combiner : a small balanced logistic regression over per-frame
                           cues (box gaps, counts, bridge ratio, same-blob, box
                           peak temp, Thermo-X3D conf), fit on train, threshold
                           tuned on val.

Outputs reports/waveshare_contact_v6_results.json.
"""

from __future__ import annotations

import json
import math
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
from scipy.ndimage import label as cc_label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
from eval_waveshare_contact_imageplane import box_gap
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
H_PX, W_PX = PROFILE.height, PROFILE.width

OUT_JSON = _ROOT / "reports" / "waveshare_contact_v6_results.json"
REFERENCE = {
    "Thermo-X3D (to beat)": dict(prec=0.484, rec=0.366, f1=0.417, far=0.024),
    "V3.0 baseline":        dict(prec=0.192, rec=0.561, f1=0.286, far=0.148),
    "V4 / V5 (best box)":   dict(prec=0.243, rec=0.659, f1=0.355, far=0.128),
}


# ---------------------------------------------------------------------------
# Thermal helpers
# ---------------------------------------------------------------------------

def _center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _box_peak(raw, b):
    x, y, w, h = (int(round(v)) for v in b)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W_PX, x + max(1, w)), min(H_PX, y + max(1, h))
    sub = raw[y0:y1, x0:x1]
    return float(sub.max()) if sub.size else float(raw.max())


def _closest_pair(boxes):
    best = (None, None, math.inf)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            g = box_gap(boxes[i], boxes[j])
            if g < best[2]:
                best = (i, j, g)
    return best


def _bridge_ratio(resid, a, b):
    """coldest point along the centre-to-centre line, relative to body warmth."""
    ca, cb = _center(a), _center(b)
    npts = max(2, int(max(abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))) + 1)
    xs = np.clip(np.round(np.linspace(ca[0], cb[0], npts)).astype(int), 0, W_PX - 1)
    ys = np.clip(np.round(np.linspace(ca[1], cb[1], npts)).astype(int), 0, H_PX - 1)
    line = resid[ys, xs]
    # body warmth = max residual at the two box centres' neighbourhoods
    body = max(resid[int(np.clip(ca[1], 0, H_PX-1)), int(np.clip(ca[0], 0, W_PX-1))],
               resid[int(np.clip(cb[1], 0, H_PX-1)), int(np.clip(cb[0], 0, W_PX-1))], 1e-3)
    return float(line.min() / body)


def _same_blob(resid, a, b, blob_thr):
    mask = resid > blob_thr
    lab, _ = cc_label(mask)
    ca, cb = _center(a), _center(b)
    la = lab[int(np.clip(ca[1], 0, H_PX-1)), int(np.clip(ca[0], 0, W_PX-1))]
    lb = lab[int(np.clip(cb[1], 0, H_PX-1)), int(np.clip(cb[0], 0, W_PX-1))]
    return la > 0 and la == lb


def _blob_thr(resid, k):
    return float(resid.mean() + k * resid.std())


# ---------------------------------------------------------------------------
# Per-camera state + router
# ---------------------------------------------------------------------------

def _cam_state(boxes, resid, raw, mode, p):
    """boxes already temp-filtered+merged. Returns 'T'/'N'/'M'/'C'."""
    n = len(boxes)
    if n == 1:
        return "M"
    if n == 0:
        return "C"
    i, j, gap = _closest_pair(boxes)
    near = gap < p["tau2"]
    if mode == "gap":
        touch = gap < p["tau"]
    elif mode == "bridge":
        touch = gap < p["tau"] and _bridge_ratio(resid, boxes[i], boxes[j]) >= p["r"]
    elif mode == "blob":
        touch = _same_blob(resid, boxes[i], boxes[j], _blob_thr(resid, p["k"]))
    else:
        raise ValueError(mode)
    if touch:
        return "T"
    return "N" if near else "C"


def _prep_boxes(cam, p, temp_filter):
    boxes = [bb for bb in cam["boxes"]]
    if temp_filter:
        boxes = [bb for bb in boxes if _box_peak(cam["raw"], bb) <= p["t_hot"]]
    return vv.merge_oversegmented(boxes)


def _router(rec, mode, p, x3dconf, temp_filter=False):
    cleaned = [_prep_boxes(c, p, temp_filter) for c in rec["cams"]]
    people = max((len(b) for b in cleaned), default=0)
    if people <= 1:
        return 0
    states = [_cam_state(cleaned[c], rec["cams"][c]["resid"], rec["cams"][c]["raw"], mode, p)
              for c in range(3)]
    if people == 2:
        votes = states.count("T"); merged = states.count("M"); clear = "C" in states
        return 1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0
    return 1 if x3dconf[id(rec)] > p["theta"] else 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def build_cache(idx, ssd, pre, labels):
    recs = []
    t0 = time.time(); n = 0
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(fis) < vv.T + 2:
            continue
        n_tr = int(len(fis) * vv.TRAIN_FRAC); n_va = int(len(fis) * (vv.TRAIN_FRAC + vv.VAL_FRAC))
        for k, fi in enumerate(fis):
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            cams = []
            for ch in CHANNELS:
                rawF = geo.get_frame(session, ch, fi)
                procF = pre[ch].predict(rawF)
                dets = ssd.predict(procF)
                cams.append({"boxes": [d.bbox for d in dets],
                             "raw": rawF.data.astype(np.float32),
                             "resid": procF.data.astype(np.float32)})
            recs.append({"split": split, "scene": scene, "fi": fi,
                         "label": int(labels[scene][fi]), "cams": cams})
            n += 1
            if n % 400 == 0:
                print(f"      ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)
    return recs


# ---------------------------------------------------------------------------
# Sweep / eval
# ---------------------------------------------------------------------------

def _cm(recs, decide):
    yt = [r["label"] for r in recs]
    yp = [decide(r) for r in recs]
    return binary_confusion_matrix(yt, yp), yp


def _tune(val, test, grid, make_decide):
    best = None
    yt = [r["label"] for r in val]
    for p in grid:
        yp = [make_decide(p)(r) for r in val]
        cm = binary_confusion_matrix(yt, yp)
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (p, cm)
    cm, yp = _cm(test, make_decide(best[0]))
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


# ---------------------------------------------------------------------------
# V6.4 learned combiner
# ---------------------------------------------------------------------------

def _features(rec, x3dconf):
    """Permutation-invariant per-frame feature vector."""
    T_HOT, K = 45.0, 1.0
    per = []
    for c in rec["cams"]:
        boxes = vv.merge_oversegmented([bb for bb in c["boxes"]
                                        if _box_peak(c["raw"], bb) <= T_HOT])
        n = len(boxes)
        if n >= 2:
            i, j, gap = _closest_pair(boxes)
            br = _bridge_ratio(c["resid"], boxes[i], boxes[j])
            sb = 1.0 if _same_blob(c["resid"], boxes[i], boxes[j], _blob_thr(c["resid"], K)) else 0.0
            pk = max(_box_peak(c["raw"], boxes[i]), _box_peak(c["raw"], boxes[j]))
        else:
            gap, br, sb = 50.0, 0.0, 0.0
            pk = _box_peak(c["raw"], boxes[0]) if n == 1 else 0.0
        per.append((n, min(gap, 50.0), min(br, 3.0), sb, pk))
    per.sort(key=lambda t: t[1])          # sort cameras by gap (invariant)
    flat = [v for t in per for v in t]
    people = max((p[0] for p in per), default=0)
    flat += [people, x3dconf[id(rec)]]
    return flat


def run_combiner(train, val, test, x3dconf):
    Xtr = np.array([_features(r, x3dconf) for r in train]); ytr = np.array([r["label"] for r in train])
    Xva = np.array([_features(r, x3dconf) for r in val]);   yva = np.array([r["label"] for r in val])
    Xte = np.array([_features(r, x3dconf) for r in test])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    for C in [0.05, 0.1, 0.3, 1.0, 3.0]:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        for thr in np.linspace(0.1, 0.9, 17):
            cm = binary_confusion_matrix(list(yva), [1 if x > thr else 0 for x in pv])
            if best is None or (cm.f1, cm.recall) > (best[2].f1, best[2].recall):
                best = (C, thr, cm, clf)
    C, thr, _, clf = best
    pte = clf.predict_proba(Xte)[:, 1]
    yp = [1 if x > thr else 0 for x in pte]
    cm = binary_confusion_matrix([r["label"] for r in test], yp)
    return {"C": C, "thr": float(thr)}, cm, yp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  V6 — thermal-aware homography-free variants")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] SSD + raw/residual cache (train+val+test) ...", flush=True)
    recs = build_cache(idx, ssd, pre, labels)
    train = [r for r in recs if r["split"] == "train"]
    val = [r for r in recs if r["split"] == "val"]
    test = [r for r in recs if r["split"] == "test"]
    print(f"      train={len(train)}  val={len(val)} (pos={sum(r['label'] for r in val)})  "
          f"test={len(test)} (pos={sum(r['label'] for r in test)})")

    print("  [2] Thermo-X3D confidences ...", flush=True)
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)
    print(f"      {len(x3dconf)} frames scored")

    results = {}
    summary = []

    # V6.1 heat-bridge
    grid = [dict(tau=t, tau2=t + e, r=r, theta=th)
            for t in (0, 2, 4, 6) for e in (0, 4, 8) for r in (0.2, 0.4, 0.6) for th in (0.6, 0.8, 0.9)]
    p, cm, yp = _tune(val, test, grid, lambda pp: (lambda r: _router(r, "bridge", pp, x3dconf)))
    results["V6.1"] = {"params": p, "rows": _per_scene(test, yp)}; summary.append(("V6.1", "heat-bridge", p, cm))

    # V6.2 temperature box rejection
    grid = [dict(t_hot=th_, tau=t, tau2=t + e, theta=th)
            for th_ in (40, 42, 45, 48) for t in (0, 2, 4, 6) for e in (0, 4, 8) for th in (0.8, 0.9)]
    p, cm, yp = _tune(val, test, grid, lambda pp: (lambda r: _router(r, "gap", pp, x3dconf, temp_filter=True)))
    results["V6.2"] = {"params": p, "rows": _per_scene(test, yp)}; summary.append(("V6.2", "temp box reject", p, cm))

    # V6.3 blob-count (same connected component)
    grid = [dict(k=k, tau=0, tau2=t2, theta=th)
            for k in (0.5, 1.0, 1.5) for t2 in (4, 8, 12) for th in (0.8, 0.9)]
    p, cm, yp = _tune(val, test, grid, lambda pp: (lambda r: _router(r, "blob", pp, x3dconf)))
    results["V6.3"] = {"params": p, "rows": _per_scene(test, yp)}; summary.append(("V6.3", "blob-count merge", p, cm))

    # V6.4 learned combiner
    p, cm, yp = run_combiner(train, val, test, x3dconf)
    results["V6.4"] = {"params": p, "rows": _per_scene(test, yp)}; summary.append(("V6.4", "learned combiner", p, cm))

    # ---- Report ----
    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    for v, desc, p, cm in summary:
        print(f"\n  {v} — {desc}   params={p}")
        print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
              f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")
        results[v]["test"] = {"acc": cm.accuracy, "prec": cm.precision, "rec": cm.recall,
                              "f1": cm.f1, "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn,
                              "fp": cm.fp, "fn": cm.fn}

    print("\n" + "=" * 80)
    print("  COMPARISON (test)")
    print("=" * 80)
    hdr = f"  {'Method':<24} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, m in REFERENCE.items():
        star = "  <-- to beat" if "to beat" in name else ""
        print(f"  {name:<24} {m['prec']:>6.1%} {m['rec']:>7.1%} {m['f1']:>7.1%} {m['far']:>7.1%}{star}")
    for v, desc, p, cm in summary:
        print(f"  {v+' '+desc:<24} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}")
    bestv = max(summary, key=lambda s: s[3].f1)
    x3df1 = REFERENCE["Thermo-X3D (to beat)"]["f1"]
    print(f"\n  Best V6: {bestv[0]} ({bestv[3].f1:.1%} F1) — "
          f"{'BEATS' if bestv[3].f1 > x3df1 else 'does NOT beat'} Thermo-X3D ({x3df1:.1%}).")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
