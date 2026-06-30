"""V10 — joint re-tune of the RAW-SSD contact pipeline (chase F1 65%).

Config D (raw SSD + residual blob + two-body merge + T1 morph) hit F1 60.8% but
reused parameters tuned for the *Tateno* pipeline (blob threshold k, tau2). The
raw SSD produces different boxes, so V10 re-tunes the whole pipeline jointly on
val:

  * SSD score threshold s   (drop low-confidence spurious person boxes)
  * blob threshold k        (residual > mean + k*std for connected components)
  * tau2                    (NEAR gap, px)
  * X3D crowd branch        (off, or OR Thermo-X3D>0.9 on 3+ people)
  * morphology              (open L_open + close L_close; plain or conf-aware)

Detector = raw SSD (checkpoints/.../Waveshare_26984_raw.thalg); blob seg on the
Tateno residual (the split shown optimal by the preprocessing ablation). Reports
the best test confusion + a per-scene error breakdown.

Outputs reports/waveshare_contact_v10_results.json.
"""

from __future__ import annotations

import json
import sys
import time
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
import eval_waveshare_contact_v8 as v8
from eval_waveshare_contact_imageplane import box_gap
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
H_PX, W_PX = PROFILE.height, PROFILE.width
RAW_CKPT = _ROOT / "checkpoints" / "mobilenet_ssd_detector" / "Waveshare_26984_raw.thalg"
OUT_JSON = _ROOT / "reports" / "waveshare_contact_v10_results.json"

SCORE_GRID = [0.3, 0.5, 0.7]
K_GRID = [0.5, 1.0, 1.5]
TAU2_GRID = [4, 8, 12]
THETA = 0.9


def _comp_counts(resid, boxes, k):
    thr = float(resid.mean() + k * resid.std())
    lab, _ = cc_label(resid > thr)
    cnt = Counter()
    for b in boxes:
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        l = lab[int(np.clip(cy, 0, H_PX - 1)), int(np.clip(cx, 0, W_PX - 1))]
        if l > 0:
            cnt[l] += 1
    return any(v >= 2 for v in cnt.values()), any(v == 2 for v in cnt.values())


def _min_gap(boxes):
    if len(boxes) < 2:
        return 1e9
    return min(box_gap(boxes[i], boxes[j])
               for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


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
                dets = ssd.predict(rawF)
                resid = pre[ch].predict(rawF).data.astype(np.float32)
                cams.append({"boxes": [(d.bbox, d.score) for d in dets], "resid": resid})
            recs.append({"split": split, "scene": scene, "fi": fi,
                         "label": int(labels[scene][fi]), "cams": cams})
            n += 1
            if n % 400 == 0:
                print(f"      ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)
    return recs


def precompute(recs):
    """Per rec/cam: for each score s -> (n, gap); for each (s,k) -> (any_merge, pair_merge)."""
    for r in recs:
        r["pc"] = []
        for c in r["cams"]:
            entry = {}
            for s in SCORE_GRID:
                boxes = vv.merge_oversegmented([bb for bb, sc in c["boxes"] if sc >= s])
                ng = (len(boxes), _min_gap(boxes))
                merges = {k: _comp_counts(c["resid"], boxes, k) for k in K_GRID}
                entry[s] = (ng, merges)
            r["pc"].append(entry)


def _state(pc_cam, s, k, tau2):
    (n, gap), merges = pc_cam[s]
    if n == 0:
        return "C"
    if n == 1:
        return "M"
    any_m, pair_m = merges[k]
    touch = any_m if n == 2 else pair_m
    if touch:
        return "T"
    return "N" if gap < tau2 else "C"


def base_preds(recs, s, k, tau2, use_x3d, x3dconf):
    pred, strong = {}, {}
    for r in recs:
        states = [_state(r["pc"][c], s, k, tau2) for c in range(3)]
        people = max(r["pc"][c][s][0][0] for c in range(3))
        votes = states.count("T"); merged = states.count("M"); clear = "C" in states
        blob = votes >= 1 and votes + merged >= 2 and not clear
        if people <= 1:
            d = 0
        elif use_x3d and people >= 3:
            d = 1 if (blob or x3dconf[id(r)] > THETA) else 0
        else:
            d = 1 if blob else 0
        pred[id(r)] = d
        strong[id(r)] = (votes >= 2) or (x3dconf[id(r)] > 0.9)
    return pred, strong


def apply_morph(recs, scenes, pred, strong, mode, lo, lc):
    out = {}
    for s, rs in scenes.items():
        arr = [pred[id(r)] for r in rs]
        if mode == "conf":
            sm = v8.morph_conf(arr, [strong[id(r)] for r in rs], lo, lc)
        else:
            sm = v8.morph(arr, lo, lc)
        for r, v in zip(rs, sm):
            out[id(r)] = v
    return out


def cm_on(recs, pred, split):
    yt = [r["label"] for r in recs if r["split"] == split]
    yp = [pred[id(r)] for r in recs if r["split"] == split]
    return binary_confusion_matrix(yt, yp)


def main():
    print("=" * 80)
    print("  V10 — joint re-tune of the RAW-SSD pipeline (target F1 65%)")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(RAW_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] raw-SSD + residual cache ...", flush=True)
    recs = build_cache(idx, ssd, pre, labels)
    print("  [2] precompute box/blob features ...", flush=True)
    precompute(recs)
    print("  [3] Thermo-X3D confidences ...", flush=True)
    x3dconf = vv.build_x3d_conf(idx, pre, labels, recs)

    scenes = defaultdict(list)
    for r in recs:
        scenes[r["scene"]].append(r)
    for s in scenes:
        scenes[s].sort(key=lambda r: r["fi"])

    print("  [4] joint sweep on val ...", flush=True)
    best = None
    for s in SCORE_GRID:
        for k in K_GRID:
            for tau2 in TAU2_GRID:
                for use_x3d in (False, True):
                    pred, strong = base_preds(recs, s, k, tau2, use_x3d, x3dconf)
                    for mode in ("plain", "conf"):
                        for lo in range(0, 5):
                            for lc in range(0, 7):
                                sm = apply_morph(recs, scenes, pred, strong, mode, lo, lc)
                                cmv = cm_on(recs, sm, "val")
                                key = (cmv.f1, cmv.recall)
                                if best is None or key > (best[0].f1, best[0].recall):
                                    best = (cmv, dict(s=s, k=k, tau2=tau2, x3d=use_x3d,
                                                      mode=mode, lo=lo, lc=lc), sm)
    cmv, p, sm = best
    cm = cm_on(recs, sm, "test")
    print(f"\n  Best params (val F1={cmv.f1:.1%}): {p}")
    print("\n" + "=" * 80)
    print("  V10 — confusion matrix (test: 41 contact / 654 no-contact)")
    print("=" * 80)
    print(f"    TP={cm.tp:<3} FN={cm.fn:<3} | FP={cm.fp:<3} TN={cm.tn:<3}   "
          f"P={cm.precision:.1%} R={cm.recall:.1%} F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}")
    print("\n  vs Config D (60.8%), V9 (55.0%), Thermo-X3D (41.7%)")

    # per-scene error breakdown
    by = defaultdict(lambda: ([], []))
    for r in recs:
        if r["split"] == "test":
            by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(sm[id(r)])
    print("\n  Per-scene errors (test):")
    rows = []
    for scene in sorted(by):
        a, b = by[scene]; c = binary_confusion_matrix(a, b)
        rows.append({"scene": scene, "tp": c.tp, "fn": c.fn, "fp": c.fp, "pos": c.tp + c.fn})
        if c.tp or c.fn or c.fp:
            print(f"    {scene:<22} TP={c.tp:<3} FN={c.fn:<3} FP={c.fp:<3} (pos={c.tp+c.fn})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"params": p, "val_f1": cmv.f1,
         "test": {"prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                  "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn,
                  "fp": cm.fp, "fn": cm.fn},
         "per_scene": rows}, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
