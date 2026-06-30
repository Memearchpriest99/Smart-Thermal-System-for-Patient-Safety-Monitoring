"""Cross-validation + motion-feature experiment for the raw-SSD contact pipeline.

Answers two things at once:
  (1) HONEST F1: 4-fold timeline cross-validation gives mean +/- std and a pooled
      estimate, instead of one noisy 40-positive split (V10 showed single-split
      tuning overfits).
  (2) DOES MOTION HELP: within the same CV, compare a logistic model WITH vs
      WITHOUT motion features, plus the config-D rule, so the gain is judged
      robustly rather than on one split.

Protocol
--------
Evaluation is restricted to the original val+test pool (the last 40% of each
session's timeline) — frames the SSD/X3D were NOT trained on, so no detector
leakage. Within that pool, each session's frames are split into 4 contiguous
quarters; fold q uses quarter q as test, quarter (q-1) as val (threshold/morph
tuning), the rest as train. Every frame is a test frame in exactly one fold, so
the pooled prediction set covers all 1108 pool frames (81 positive).

Base signal = config D (raw SSD boxes + residual two-body blob-merge, k=1,
tau2=8). Motion features: frame-difference energy in the warm region + its
window mean + merge run-length (sustained vs transient).

Outputs reports/waveshare_contact_cv_motion_results.json.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
OUT_JSON = _ROOT / "reports" / "waveshare_contact_cv_motion_results.json"
K, TAU2 = 1.0, 8.0
N_FOLDS = 4


def _comp(resid, boxes, k):
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
        return 50.0
    return min(min(box_gap(boxes[i], boxes[j]), 50.0)
               for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


def build_pool(idx, ssd, pre, labels):
    """val+test frames only, per scene, in timeline order."""
    scenes = {}
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
        n_tr = int(len(fis) * vv.TRAIN_FRAC)
        pool = fis[n_tr:]                       # last 40% (val+test) — no SSD leakage
        recs = []
        for fi in pool:
            cams = []
            for ch in CHANNELS:
                rawF = geo.get_frame(session, ch, fi)
                dets = ssd.predict(rawF)
                resid = pre[ch].predict(rawF).data.astype(np.float32)
                cams.append({"boxes": [d.bbox for d in dets], "resid": resid})
            recs.append({"scene": scene, "fi": fi, "label": int(labels[scene][fi]), "cams": cams})
            n += 1
            if n % 300 == 0:
                print(f"      ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)
        if recs:
            scenes[scene] = recs
    return scenes


def precompute(scenes, x3dconf):
    for scene, recs in scenes.items():
        run = 0
        for r in recs:
            cleaned = [vv.merge_oversegmented([bb for bb in c["boxes"]]) for c in r["cams"]]
            people = max((len(b) for b in cleaned), default=0)
            states = []
            for c, boxes in zip(r["cams"], cleaned):
                n = len(boxes)
                if n == 0:
                    st = "C"
                elif n == 1:
                    st = "M"
                else:
                    any_m, pair_m = _comp(c["resid"], boxes, K)
                    touch = any_m if n == 2 else pair_m
                    st = "T" if touch else ("N" if _min_gap(boxes) < TAU2 else "C")
                states.append(st)
            votes = states.count("T"); merged = states.count("M"); clear = "C" in states
            blob = votes >= 1 and votes + merged >= 2 and not clear
            base = 0 if people <= 1 else (1 if blob else 0)
            run = run + 1 if base == 1 else 0
            r.update(base=base, votes=votes, merged=merged, people=people,
                     x3d=float(x3dconf[id(r)]), merge_run=run)
        # second pass for motion (needs previous frame per camera)
        for i, r in enumerate(recs):
            if i == 0:
                r["motion"] = 0.0
                continue
            prev = recs[i - 1]
            es = []
            for c, pc in zip(r["cams"], prev["cams"]):
                R, P = c["resid"], pc["resid"]
                thr = float(R.mean() + R.std())
                mask = (R > thr) | (P > thr)
                if mask.any():
                    es.append(float(np.abs(R - P)[mask].mean()))
            r["motion"] = float(np.mean(es)) if es else 0.0
        # window-mean motion + min_gap across cameras (recompute proper min over cams)
        for i, r in enumerate(recs):
            lo, hi = max(0, i - 2), min(len(recs), i + 3)
            r["motion_w"] = float(np.mean([recs[j]["motion"] for j in range(lo, hi)]))
            mg = 50.0
            for c in r["cams"]:
                b = vv.merge_oversegmented([bb for bb in c["boxes"]])
                mg = min(mg, _min_gap(b))
            r["min_gap"] = mg


FEATS_NOMOT = ["base", "x3d", "votes", "merged", "people", "min_gap"]
FEATS_MOT = FEATS_NOMOT + ["motion", "motion_w", "merge_run"]


def fold_assign(recs):
    n = len(recs)
    for rank, r in enumerate(recs):
        r["q"] = min(N_FOLDS - 1, rank * N_FOLDS // n)


def eval_rule(scenes, test_q, val_q):
    """config-D base + morphology tuned on val quarter, scored on test quarter."""
    best = None
    for lo in range(0, 5):
        for lc in range(0, 7):
            yt, yp = [], []
            for scene, recs in scenes.items():
                arr = [r["base"] for r in recs]
                sm = v8.morph(arr, lo, lc)
                for r, v in zip(recs, sm):
                    if r["q"] == val_q:
                        yt.append(r["label"]); yp.append(v)
            cm = binary_confusion_matrix(yt, yp)
            if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cm)
    lo, lc = best[0]
    pred = {}
    for scene, recs in scenes.items():
        sm = v8.morph([r["base"] for r in recs], lo, lc)
        for r, v in zip(recs, sm):
            pred[id(r)] = v
    return pred


def eval_logreg(scenes, test_q, val_q, feats):
    Xtr, ytr, Xva, yva = [], [], [], []
    allr = [r for recs in scenes.values() for r in recs]
    for r in allr:
        x = [r[f] for f in feats]
        if r["q"] == test_q:
            continue
        if r["q"] == val_q:
            Xva.append(x); yva.append(r["label"])
        else:
            Xtr.append(x); ytr.append(r["label"])
    Xtr, Xva = np.array(Xtr), np.array(Xva)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    best = None
    for C in (0.05, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        pv = clf.predict_proba(Xva)[:, 1]
        for thr in np.linspace(0.2, 0.9, 15):
            cm = binary_confusion_matrix(list(yva), [1 if p > thr else 0 for p in pv])
            if best is None or (cm.f1, cm.recall) > (best[2].f1, best[2].recall):
                best = (C, thr, cm, clf, sc)
    C, thr, _, clf, sc = best
    pred = {}
    for r in allr:
        x = sc.transform([[r[f] for f in feats]])
        pred[id(r)] = 1 if clf.predict_proba(x)[0, 1] > thr else 0
    return pred


def cv(scenes, evaluator):
    """Run N_FOLDS; return (per-fold F1 list, pooled cm)."""
    fold_f1 = []
    pooled_yt, pooled_yp = [], []
    for q in range(N_FOLDS):
        val_q = (q - 1) % N_FOLDS
        pred = evaluator(scenes, q, val_q)
        yt, yp = [], []
        for recs in scenes.values():
            for r in recs:
                if r["q"] == q:
                    yt.append(r["label"]); yp.append(pred[id(r)])
        cm = binary_confusion_matrix(yt, yp)
        fold_f1.append(cm.f1)
        pooled_yt += yt; pooled_yp += yp
    return fold_f1, binary_confusion_matrix(pooled_yt, pooled_yp)


def main():
    print("=" * 80)
    print("  CROSS-VALIDATION + MOTION FEATURES (raw-SSD pipeline)")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    ssd = MobileNetSSDDetector.load(RAW_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  [1] build val+test pool (raw SSD) ...", flush=True)
    scenes = build_pool(idx, ssd, pre, labels)
    flat = [r for recs in scenes.values() for r in recs]
    print(f"      pool: {len(flat)} frames, {sum(r['label'] for r in flat)} positive, "
          f"{len(scenes)} scenes")

    print("  [2] Thermo-X3D confidences ...", flush=True)
    # build_x3d_conf keys by id(rec); replay full sessions for warm buffer
    x3dconf = vv.build_x3d_conf(idx, pre, labels, flat)

    print("  [3] precompute base + motion features ...", flush=True)
    precompute(scenes, x3dconf)
    for recs in scenes.values():
        fold_assign(recs)

    print("  [4] cross-validation (4 folds) ...", flush=True)
    detectors = {
        "config-D rule (+morph)": lambda s, tq, vq: eval_rule(s, tq, vq),
        "logreg  (no motion)":    lambda s, tq, vq: eval_logreg(s, tq, vq, FEATS_NOMOT),
        "logreg  (+ motion)":     lambda s, tq, vq: eval_logreg(s, tq, vq, FEATS_MOT),
    }
    results = {}
    print("\n" + "=" * 80)
    print(f"  {'Detector':<26} {'fold F1s':<28} {'mean±std':>14} {'pooled F1':>10} {'pooled P/R/FAR':>22}")
    print("  " + "-" * 100)
    for name, ev in detectors.items():
        f1s, pcm = cv(scenes, ev)
        mean, std = float(np.mean(f1s)), float(np.std(f1s))
        fstr = " ".join(f"{x:.0%}" for x in f1s)
        print(f"  {name:<26} {fstr:<28} {mean:>6.1%} ± {std:>4.1%}  {pcm.f1:>9.1%}  "
              f"{pcm.precision:>5.1%}/{pcm.recall:.1%}/{pcm.false_alarm_rate:.1%}")
        results[name] = {"fold_f1": f1s, "mean": mean, "std": std,
                         "pooled": {"f1": pcm.f1, "prec": pcm.precision, "rec": pcm.recall,
                                    "far": pcm.false_alarm_rate, "tp": pcm.tp, "tn": pcm.tn,
                                    "fp": pcm.fp, "fn": pcm.fn}}

    print("\n  Notes:")
    print("   * Single-split numbers earlier: config D 60.8%, V9 55.0% (test=695 frames).")
    print("   * CV pooled covers all 1108 pool frames (81 pos), each tested once — no SSD leakage.")
    dm = results["logreg  (+ motion)"]["mean"]; dn = results["logreg  (no motion)"]["mean"]
    print(f"   * Motion effect (CV mean F1): {dn:.1%} -> {dm:.1%}  (Δ {100*(dm-dn):+.1f}pt)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
