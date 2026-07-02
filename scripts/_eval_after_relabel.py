"""Full-dataset contact benchmark AFTER re-annotation of the 3 suspect videos.

Scores config-D, the original Thermo-X3D and the clean-trained Thermo-X3D over
every labelled frame (17 scenes) with the corrected labels, pooled, at tol=0
and +-2. CAVEAT: on the 3 re-annotated videos, 70% of labels came from
config-D/clean-X3D consensus, so both models are favoured there by
construction; the 167 human-reviewed frames are unbiased.

Outputs reports/post_relabel_benchmark.json.
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
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

RELABELED = {"3pp_surprise", "3ppl_dance", "the_more_the_merrier"}
LO, LC = 2, 7
X3D_OLD = (_ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984.thalg", 0.75)
X3D_CLEAN = (_ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_clean.thalg", 0.1)

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
x3d_old = ThermoX3DDetector.load(X3D_OLD[0])
x3d_clean = ThermoX3DDetector.load(X3D_CLEAN[0])

print("scoring all scenes with corrected labels ...", flush=True)
scenes = {}
t0 = time.time(); n = 0
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    x3d_old.reset(); x3d_clean.reset()
    labs, d_base, c_old, c_clean = [], [], [], []
    for fi in fis:
        raws = [geo.get_frame(s, ch, fi) for ch in geo.CHANNELS]
        resids = tuple(pre[ch].predict(raws[c]) for c, ch in enumerate(geo.CHANNELS))
        dets_r = [ssd_raw.predict(rw) for rw in raws]
        d_base.append(ab.v9core_decision(
            {"cams": [{"boxes": [d.bbox for d in dets_r[c]],
                       "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid"))
        c_old.append(float(x3d_old.predict(resids).confidence))
        c_clean.append(float(x3d_clean.predict(resids).confidence))
        labs.append(int(labels[scene][fi]))
        n += 1
        if n % 300 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    scenes[scene] = {
        "labels": labs,
        "config_D": list(v8.morph(d_base, LO, LC)),
        "x3d_old": [1 if c > X3D_OLD[1] else 0 for c in c_old],
        "x3d_clean": [1 if c > X3D_CLEAN[1] else 0 for c in c_clean],
    }

def pooled(algo, tol, subset=None):
    tp = fn = fp = tn = 0
    for sc, d in scenes.items():
        if subset and sc not in subset: continue
        labs, yp = d["labels"], d[algo]
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

ALGOS = ["config_D", "x3d_old", "x3d_clean"]
npos = sum(sum(d["labels"]) for d in scenes.values())
ntot = sum(len(d["labels"]) for d in scenes.values())
print(f"\nFull dataset, corrected labels: {len(scenes)} scenes, {ntot} frames, {npos} positive")
print("=" * 92)
print(f"  {'algorithm':<12} | {'P t0':>6} {'R t0':>6} {'F1 t0':>6} {'FAR':>5} | {'P ±2':>6} {'R ±2':>6} {'F1 ±2':>6} {'FAR':>5}")
print("  " + "-" * 88)
out = {"n_scenes": len(scenes), "n_frames": ntot, "n_pos": npos, "pooled": {}, "per_scene_f1": {}}
for a in ALGOS:
    m0, m2 = pooled(a, 0), pooled(a, 2)
    out["pooled"][a] = {"tol0": m0, "tol2": m2}
    print(f"  {a:<12} | {m0['prec']:>6.1%} {m0['rec']:>6.1%} {m0['f1']:>6.1%} {m0['far']:>5.1%} | "
          f"{m2['prec']:>6.1%} {m2['rec']:>6.1%} {m2['f1']:>6.1%} {m2['far']:>5.1%}")

print(f"\n  Per-scene F1 (tol=0), contact scenes:")
print(f"  {'scene':<24} {'pos':>4} " + " ".join(f"{a:>10}" for a in ALGOS))
for sc in sorted(scenes):
    labs = scenes[sc]["labels"]
    if not any(labs): continue
    mark = " *" if sc in RELABELED else "  "
    row = f"  {sc:<22}{mark} {sum(labs):>4} "
    pf = {}
    for a in ALGOS:
        cm = binary_confusion_matrix(labs, scenes[sc][a])
        pf[a] = cm.f1
        row += f" {cm.f1:>9.1%}"
    out["per_scene_f1"][sc] = pf
    print(row)
print("  (* = re-annotated via consensus + human review — favours config-D/x3d_clean by construction)")

out_path = _ROOT / "reports" / "post_relabel_benchmark.json"
out_path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {out_path}")
