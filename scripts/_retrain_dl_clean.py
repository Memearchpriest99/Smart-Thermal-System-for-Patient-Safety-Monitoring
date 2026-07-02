"""Retrain the DL contact detectors on CLEAN labels only, then re-benchmark.

The 3 suspect videos (the_more_the_merrier, 3ppl_dance = never annotated;
3pp_surprise = block-labeled) are excluded from BOTH training and evaluation.
Training procedure is identical to eval_waveshare_contact_dl.py (same splits,
epochs, class weighting, val threshold sweep) — only the scene set changes.
Checkpoints go to new *_clean paths; the polluted ckpts are kept for provenance.

Outputs reports/dl_clean_retrain_results.json:
  - honest val-tuned / test-scored numbers per detector (from the dl protocol)
  - pooled full-replay benchmark over all clean frames at tol=0 / tol=+-2,
    comparable to reports/clean_label_benchmark.json (config-D reference row).
Also cross-checks the reloaded checkpoint against the in-memory model to catch
the degenerate-save problem seen with mv_stgcn/_default.thalg.
"""
import json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np

import eval_waveshare_contact as geo
import eval_waveshare_contact_dl as dl
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

SUSPECT = {"the_more_the_merrier", "3ppl_dance", "3pp_surprise"}

# retarget checkpoint paths BEFORE training (run_* saves to these module globals)
dl.X3D_CKPT = _ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_clean.thalg"
dl.STGCN_CKPT = _ROOT / "checkpoints" / "mv_stgcn_detector" / "_clean.thalg"

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
pre = geo.fit_preprocessors(idx)
ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
H, _info, _n = geo.calibrate_homography(idx)

print("building per-session sequences (clean scenes only) ...", flush=True)
seqs = dl.build_sequences(idx, pre, ssd)
seqs = {k: v for k, v in seqs.items() if k not in SUSPECT}
npos = sum(r["label"] for rs in seqs.values() for r in rs)
ntr = sum(1 for rs in seqs.values() for r in rs if r["split"] == "train")
ptr = sum(r["label"] for rs in seqs.values() for r in rs if r["split"] == "train")
print(f"  {len(seqs)} scenes, {sum(len(v) for v in seqs.values())} frames "
      f"({npos} pos) | train {ntr} ({ptr} pos)", flush=True)

res_x3d = dl.run_thermo_x3d(seqs)
res_stgcn = dl.run_mv_stgcn(seqs, H)

# ---------------------------------------------------------------------------
# Pooled full-replay benchmark with the RELOADED clean checkpoints
# ---------------------------------------------------------------------------
x3d = ThermoX3DDetector.load(dl.X3D_CKPT)
stgcn = MVSTGCNDetector.load(dl.STGCN_CKPT)
th_x3d = res_x3d["best_threshold"]
th_stgcn = res_stgcn["best_threshold"]
print(f"\nfull replay with reloaded ckpts (th_x3d={th_x3d}, th_stgcn={th_stgcn}) ...", flush=True)

conf = {}   # scene -> {"x3d": [...], "stgcn": [...]}
for scene, recs in seqs.items():
    x3d.reset(); stgcn.reset()
    cx, cs = [], []
    for r in recs:
        cx.append(float(x3d.predict(r["triplet"]).confidence))
        cs.append(float(stgcn.predict(r["triplet"], detections=r["dets"]).confidence))
    conf[scene] = {"x3d": cx, "stgcn": cs}

# degenerate-save check: reloaded confidences should not be ~constant
for name in ("x3d", "stgcn"):
    allc = np.array([c for sc in conf.values() for c in sc[name]])
    print(f"  reloaded {name}: conf min={allc.min():.3f} max={allc.max():.3f} "
          f"std={allc.std():.3f}{'   << DEGENERATE?' if allc.std() < 1e-3 else ''}")

def pooled(name, th, tol):
    tp = fn = fp = tn = 0
    for scene, recs in seqs.items():
        labs = [r["label"] for r in recs]
        yp = [1 if c > th else 0 for c in conf[scene][name]]
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

print("\n" + "=" * 88)
print("  pooled full-replay (all clean frames; train frames included — diagnostic only)")
print(f"  {'model':<14} | {'TP':>4} {'FN':>4} {'FP':>4} | {'P t0':>6} {'R t0':>6} {'F1 t0':>6} | {'P ±2':>6} {'R ±2':>6} {'F1 ±2':>6} {'FAR':>5}")
print("  " + "-" * 86)
out_bench = {}
for name, th in (("x3d", th_x3d), ("stgcn", th_stgcn)):
    m0, m2 = pooled(name, th, 0), pooled(name, th, 2)
    out_bench[name] = {"threshold": th, "tol0": m0, "tol2": m2}
    print(f"  {name:<14} | {m0['tp']:>4} {m0['fn']:>4} {m0['fp']:>4} | "
          f"{m0['prec']:>6.1%} {m0['rec']:>6.1%} {m0['f1']:>6.1%} | "
          f"{m2['prec']:>6.1%} {m2['rec']:>6.1%} {m2['f1']:>6.1%} {m2['far']:>5.1%}")

# per-scene F1 (tol=0) on contact scenes
contact_scenes = [sc for sc, rs in seqs.items() if any(r["label"] for r in rs)]
from thermal_algorithms.training.metrics import binary_confusion_matrix
print(f"\n  per-scene F1 (tol=0): {contact_scenes}")
for sc in sorted(contact_scenes):
    labs = [r["label"] for r in seqs[sc]]
    row = f"  {sc:<20}"
    for name, th in (("x3d", th_x3d), ("stgcn", th_stgcn)):
        cm = binary_confusion_matrix(labs, [1 if c > th else 0 for c in conf[sc][name]])
        row += f"  {name}={cm.f1:.1%}"
    print(row)

out = {"suspect_excluded": sorted(SUSPECT),
       "protocol": {"train_frac": dl.TRAIN_FRAC, "val_frac": dl.VAL_FRAC, "T": dl.T},
       "ThermoX3D_clean": res_x3d, "MVSTGCN_clean": res_stgcn,
       "full_replay_pooled": out_bench}
out_path = _ROOT / "reports" / "dl_clean_retrain_results.json"
out_path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {out_path}")
