"""Thermo-X3D with a SHORT temporal window (T=5, ~0.6s at 8Hz) on corrected labels.

Motivation: the corrected labels have dense contact transitions inside crowd
scenes; a T=16 (2s) window spans contradictory ground truth relative to its
last-frame target. T=5 matches label granularity and yields more windows.
Same protocol otherwise. Checkpoint -> Waveshare_26984_relabel_T5.thalg.
"""
import json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

import eval_waveshare_contact as geo
import eval_waveshare_contact_dl as dl
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

dl.T = 5
dl.X3D_CKPT = _ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_relabel_T5.thalg"

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
pre = geo.fit_preprocessors(idx)
ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)

print(f"building sequences (ALL scenes, corrected labels, T={dl.T}) ...", flush=True)
seqs = dl.build_sequences(idx, pre, ssd)
npos = sum(r["label"] for rs in seqs.values() for r in rs)
print(f"  {len(seqs)} scenes, {sum(len(v) for v in seqs.values())} frames ({npos} pos)", flush=True)

res = dl.run_thermo_x3d(seqs)

x3d = ThermoX3DDetector.load(dl.X3D_CKPT)
th = res["best_threshold"]
print(f"\nfull replay with reloaded ckpt (th={th}) ...", flush=True)
scenes = {}
for scene, recs in seqs.items():
    x3d.reset()
    confs = [float(x3d.predict(r["triplet"]).confidence) for r in recs]
    scenes[scene] = ([r["label"] for r in recs], confs)

def pooled(th_, tol):
    tp = fn = fp = tn = 0
    for labs, confs in scenes.values():
        yp = [1 if c > th_ else 0 for c in confs]
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

m0, m2 = pooled(th, 0), pooled(th, 2)
print(f"  pooled tol=0 : P={m0['prec']:.1%} R={m0['rec']:.1%} F1={m0['f1']:.1%} FAR={m0['far']:.1%}")
print(f"  pooled tol=±2: P={m2['prec']:.1%} R={m2['rec']:.1%} F1={m2['f1']:.1%} FAR={m2['far']:.1%}")
print("\n  per-scene F1 (tol=0), contact scenes:")
per_scene = {}
for sc in sorted(scenes):
    labs, confs = scenes[sc]
    if not any(labs): continue
    yp = [1 if c > th else 0 for c in confs]
    cm = binary_confusion_matrix(labs, yp)
    per_scene[sc] = cm.f1
    print(f"    {sc:<24} pos={sum(labs):>3}  F1={cm.f1:.1%}  (TP={cm.tp} FN={cm.fn} FP={cm.fp})")

out = {"T": dl.T, "threshold": th, "honest": res,
       "full_replay": {"tol0": m0, "tol2": m2}, "per_scene_f1": per_scene,
       "confidences": {sc: {"labels": labs, "confs": confs} for sc, (labs, confs) in scenes.items()}}
out_path = _ROOT / "reports" / "x3d_relabel_T5_results.json"
out_path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {out_path}")
