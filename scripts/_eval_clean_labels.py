"""Benchmark ALL contact detectors on the CLEAN-LABEL subset only.

The 2026-07-02 label audit (_analyze_label_quality.py) found three videos with
unusable touch labels: the_more_the_merrier + 3ppl_dance (never annotated,
all-zero) and 3pp_surprise (block-labeled ~96% positive). This script re-scores
every algorithm family on the remaining 14 videos, pooled over ALL their
labeled frames, at tol=0 and with a +-2 annotated-frame boundary tolerance.

Algorithms (existing checkpoints / tuned params, NO retraining):
  geometric   Tateno-SSD -> foot-points -> self-calib H -> fusion  (min_src=2, eps=15, delta=18)
  mv_stgcn    ckpt mv_stgcn_detector/_default.thalg, th=0.40
  thermo_x3d  ckpt thermo_x3d_detector/Waveshare_26984.thalg, th=0.75
  config_A    Tateno-SSD boxes + resid blob + 2-body + morph(2,7)
  config_D    raw-SSD boxes    + resid blob + 2-body + morph(2,7)   <- champion

Caveats: SSD ckpts saw ~60-80%% of these frames at train time (whole-dataset
diagnostic, same as _configd_all.py); the DL ckpts were trained with the
polluted labels still in their train split, so their numbers are a lower bound.
"""
import json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

SUSPECT = {"the_more_the_merrier", "3ppl_dance", "3pp_surprise"}
EPS_PX, MIN_SRC, DELTA_PX = 15.0, 2, 18.0
X3D_TH, STGCN_TH = 0.75, 0.40
LO, LC = 2, 7

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)
ssd_tat = MobileNetSSDDetector.load(geo.SSD_CKPT)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
H, _info, _n = geo.calibrate_homography(idx)
x3d = ThermoX3DDetector.load(_ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984.thalg")
stgcn = MVSTGCNDetector.load(_ROOT / "checkpoints" / "mv_stgcn_detector" / "_default.thalg")

print("scoring clean scenes ...", flush=True)
scenes = {}
t0 = time.time(); n = 0
for scene in sorted(labels):
    if scene in SUSPECT: continue
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    x3d.reset(); stgcn.reset()
    recs = []
    for fi in fis:
        raws = [geo.get_frame(s, ch, fi) for ch in geo.CHANNELS]
        resids = tuple(pre[ch].predict(raws[c]) for c, ch in enumerate(geo.CHANNELS))
        dets_t = tuple(ssd_tat.predict(r) for r in resids)
        dets_r = [ssd_raw.predict(rw) for rw in raws]
        actors = geo.actors_from_dets(list(dets_t), H, epsilon=EPS_PX, min_sources=MIN_SRC)
        rec = {
            "fi": fi, "label": int(labels[scene][fi]),
            "geo": geo._contact_pred(actors, DELTA_PX),
            "x3d": float(x3d.predict(resids).confidence),
            "stgcn": float(stgcn.predict(resids, detections=dets_t).confidence),
            "a_base": ab.v9core_decision(
                {"cams": [{"boxes": [d.bbox for d in dets_t[c]],
                           "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid"),
            "d_base": ab.v9core_decision(
                {"cams": [{"boxes": [d.bbox for d in dets_r[c]],
                           "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid"),
        }
        recs.append(rec); n += 1
        if n % 300 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    scenes[scene] = recs

# ---- per-algorithm per-scene prediction streams -------------------------------
def streams(recs):
    return {
        "geometric":  [r["geo"] for r in recs],
        "mv_stgcn":   [1 if r["stgcn"] > STGCN_TH else 0 for r in recs],
        "thermo_x3d": [1 if r["x3d"] > X3D_TH else 0 for r in recs],
        "config_A":   v8.morph([r["a_base"] for r in recs], LO, LC),
        "config_D":   v8.morph([r["d_base"] for r in recs], LO, LC),
    }

ALGOS = ["geometric", "mv_stgcn", "thermo_x3d", "config_A", "config_D"]
per_scene_preds = {sc: streams(rs) for sc, rs in scenes.items()}

def pooled(algo, tol):
    tp = fn = fp = tn = 0
    for sc, rs in scenes.items():
        labs = [r["label"] for r in rs]
        yp = per_scene_preds[sc][algo]
        for i, (l, p) in enumerate(zip(labs, yp)):
            if tol and p != l:
                lo_, hi_ = max(0, i - tol), min(len(labs), i + tol + 1)
                if p in labs[lo_:hi_]: l = p
            tp += (p == 1 and l == 1); fn += (p == 0 and l == 1)
            fp += (p == 1 and l == 0); tn += (p == 0 and l == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, prec=prec, rec=rec, f1=f1, far=far)

npos = sum(r["label"] for rs in scenes.values() for r in rs)
ntot = sum(len(rs) for rs in scenes.values())
print(f"\nClean subset: {len(scenes)} scenes, {ntot} frames, {npos} positive")
print("=" * 88)
print(f"  {'algorithm':<12} | {'tol=0':^38} | {'tol=+-2':^30}")
print(f"  {'':<12} | {'TP':>4} {'FN':>4} {'FP':>4} {'P':>6} {'R':>6} {'F1':>6} {'FAR':>5} | {'P':>6} {'R':>6} {'F1':>6} {'FAR':>5}")
print("  " + "-" * 86)
out = {"n_scenes": len(scenes), "n_frames": ntot, "n_pos": npos, "algorithms": {}}
for a in ALGOS:
    m0, m2 = pooled(a, 0), pooled(a, 2)
    out["algorithms"][a] = {"tol0": m0, "tol2": m2}
    print(f"  {a:<12} | {m0['tp']:>4} {m0['fn']:>4} {m0['fp']:>4} "
          f"{m0['prec']:>6.1%} {m0['rec']:>6.1%} {m0['f1']:>6.1%} {m0['far']:>5.1%} | "
          f"{m2['prec']:>6.1%} {m2['rec']:>6.1%} {m2['f1']:>6.1%} {m2['far']:>5.1%}")

# ---- per-scene F1 on contact scenes (tol=0) ------------------------------------
contact_scenes = [sc for sc, rs in scenes.items() if any(r["label"] for r in rs)]
print(f"\n  Per-scene F1 (tol=0) on contact scenes: {contact_scenes}")
print(f"  {'scene':<20} " + " ".join(f"{a:>11}" for a in ALGOS))
for sc in contact_scenes:
    labs = [r["label"] for r in scenes[sc]]
    row = f"  {sc:<20} "
    for a in ALGOS:
        cm = binary_confusion_matrix(labs, list(per_scene_preds[sc][a]))
        row += f" {cm.f1:>10.1%}"
    print(row)
# FP counts on negative-only scenes
print(f"\n  FP count on all-negative scenes (tol=0):")
neg_scenes = [sc for sc in scenes if sc not in contact_scenes]
print(f"  {'scene':<20} " + " ".join(f"{a:>11}" for a in ALGOS))
for sc in neg_scenes:
    labs = [r["label"] for r in scenes[sc]]
    row = f"  {sc:<20} "
    for a in ALGOS:
        cm = binary_confusion_matrix(labs, list(per_scene_preds[sc][a]))
        row += f" {cm.fp:>11d}"
    print(row)

out_path = _ROOT / "reports" / "clean_label_benchmark.json"
out_path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {out_path}")
