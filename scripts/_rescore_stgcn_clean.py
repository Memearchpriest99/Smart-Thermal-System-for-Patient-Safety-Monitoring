"""Re-score MV-STGCN (clean-trained, REPAIRED checkpoint) over the clean subset.

The homography-persistence bug meant the earlier replay ran with a dead fusion
front-end. Checkpoint now carries H; this replays all clean scenes and reports
pooled tol=0 / +-2 metrics, plus honest val/test numbers at the tuned threshold.
Updates full_replay_pooled.stgcn in reports/dl_clean_retrain_results.json.
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
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

SUSPECT = {"the_more_the_merrier", "3ppl_dance", "3pp_surprise"}
CKPT = _ROOT / "checkpoints" / "mv_stgcn_detector" / "_clean.thalg"

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)
ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
stgcn = MVSTGCNDetector.load(CKPT)
assert stgcn.homography is not None, "checkpoint still missing homography"

print("replaying clean scenes with repaired STGCN ...", flush=True)
scenes = {}   # scene -> (labels, confs, splits)
t0 = time.time(); n = 0
for scene in sorted(labels):
    if scene in SUSPECT: continue
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    n_tr = int(len(fis) * dl.TRAIN_FRAC)
    n_va = int(len(fis) * (dl.TRAIN_FRAC + dl.VAL_FRAC))
    stgcn.reset()
    labs, confs, splits = [], [], []
    for k, fi in enumerate(fis):
        resids = tuple(pre[ch].predict(geo.get_frame(s, ch, fi)) for ch in geo.CHANNELS)
        dets = tuple(ssd.predict(r) for r in resids)
        confs.append(float(stgcn.predict(resids, detections=dets).confidence))
        labs.append(int(labels[scene][fi]))
        splits.append("train" if k < n_tr else ("val" if k < n_va else "test"))
        n += 1
        if n % 300 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    scenes[scene] = (labs, confs, splits)

allc = np.array([c for _, cs, _ in scenes.values() for c in cs])
print(f"  conf: min={allc.min():.3f} max={allc.max():.3f} std={allc.std():.3f}")

# honest protocol: tune threshold on val, score test
scored = [(sp, sc, l, c) for sc, (ls, cs, sps) in scenes.items()
          for l, c, sp in zip(ls, cs, sps)]
best_th, sweep, rows, agg = dl.sweep_threshold(scored)
dl.report("MV-STGCN (repaired ckpt, clean labels)", best_th, rows, agg)

def pooled(th, tol):
    tp = fn = fp = tn = 0
    for sc, (labs, confs, _) in scenes.items():
        yp = [1 if c > th else 0 for c in confs]
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

m0, m2 = pooled(best_th, 0), pooled(best_th, 2)
print(f"\n  pooled full-replay (th={best_th}):")
print(f"    tol=0 : TP={m0['tp']} FN={m0['fn']} FP={m0['fp']} P={m0['prec']:.1%} R={m0['rec']:.1%} F1={m0['f1']:.1%} FAR={m0['far']:.1%}")
print(f"    tol=±2: TP={m2['tp']} FN={m2['fn']} FP={m2['fp']} P={m2['prec']:.1%} R={m2['rec']:.1%} F1={m2['f1']:.1%} FAR={m2['far']:.1%}")

for sc in sorted(scenes):
    labs, confs, _ = scenes[sc]
    if not any(labs): continue
    cm = binary_confusion_matrix(labs, [1 if c > best_th else 0 for c in confs])
    print(f"    {sc:<20} F1={cm.f1:.1%} (TP={cm.tp} FN={cm.fn} FP={cm.fp})")

out_path = _ROOT / "reports" / "dl_clean_retrain_results.json"
out = json.loads(out_path.read_text())
out["full_replay_pooled"]["stgcn"] = {"threshold": best_th, "tol0": m0, "tol2": m2,
                                      "note": "repaired ckpt (homography persistence fix)"}
out["MVSTGCN_clean_repaired"] = {"best_threshold": best_th,
                                 "test": {"aggregate": {"tp": agg.tp, "fn": agg.fn, "fp": agg.fp,
                                                        "tn": agg.tn, "prec": agg.precision,
                                                        "rec": agg.recall, "f1": agg.f1,
                                                        "far": agg.false_alarm_rate},
                                          "per_scene": rows}}
out_path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {out_path}")
