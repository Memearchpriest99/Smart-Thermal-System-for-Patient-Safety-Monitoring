"""Does session-median background cost anything at home?

Re-runs the post-relabel benchmark (config-D + X3D-T5) over all labelled scenes
with the PER-SCENE MEDIAN background instead of the empty_room-fit Tateno
background. If home numbers hold, session-median becomes the standard
everywhere (it is already mandatory cross-room — proven on 2men_clash).

Includes 2men_clash (human-labelled held-out). Reference (empty_room bg):
  config-D pooled ±2: P65.9 R94.4 F1 77.6 FAR 4.5   (17 scenes, no 2men_clash)
  X3D-T5 pooled ±2:   P38.5 R99.1 F1 55.5 FAR 21.8

Outputs reports/home_sessionbg_benchmark.json.
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
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

LO, LC = 2, 7
X3D_T5 = (_ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_relabel_T5.thalg", 0.2)

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
x3d = ThermoX3DDetector.load(X3D_T5[0])

BG_PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0   # 50 = median

def session_pre(s, fis):
    pre = {}
    for ch in geo.CHANNELS:
        stack = np.stack([geo.get_frame(s, ch, fi).data for fi in fis], 0)
        # low percentile biases toward the cool background — people are warm,
        # so this resists absorbing static occupants on short clips
        med = np.percentile(stack, BG_PCT, axis=0)
        pre[ch] = TatenoPipeline(geo.PROFILE).fit([Frame(data=med, timestamp=0.0, camera_id=ch)])
    return pre

print("scoring all labelled scenes with SESSION-MEDIAN background ...", flush=True)
scenes = {}
t0 = time.time(); n = 0
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    pre = session_pre(s, fis)
    x3d.reset()
    labs, d_base, xconf = [], [], []
    for fi in fis:
        raws = [geo.get_frame(s, ch, fi) for ch in geo.CHANNELS]
        resids = tuple(pre[ch].predict(raws[c]) for c, ch in enumerate(geo.CHANNELS))
        dets_r = [ssd_raw.predict(rw) for rw in raws]
        d_base.append(ab.v9core_decision(
            {"cams": [{"boxes": [d.bbox for d in dets_r[c]],
                       "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid"))
        xconf.append(float(x3d.predict(resids).confidence))
        labs.append(int(labels[scene][fi]))
        n += 1
        if n % 300 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    scenes[scene] = {"labels": labs,
                     "config_D": list(v8.morph(d_base, LO, LC)),
                     "x3d_T5": [1 if c > X3D_T5[1] else 0 for c in xconf]}

def pooled(algo, tol, exclude=()):
    tp = fn = fp = tn = 0
    for sc, d in scenes.items():
        if sc in exclude: continue
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

REF = {"config_D": {"f1t0": 0.609, "f1t2": 0.776, "far": 0.045},
       "x3d_T5":   {"f1t0": 0.405, "f1t2": 0.555, "far": 0.218}}
print("\n" + "=" * 96)
print("  session-median background, all labelled scenes (2men_clash separated)")
print(f"  {'model':<10} {'set':<22} | {'P t0':>6} {'R t0':>6} {'F1 t0':>6} | {'P ±2':>6} {'R ±2':>6} {'F1 ±2':>6} {'FAR':>5} | ref F1±2/FAR (empty-bg)")
print("  " + "-" * 92)
out = {}
for a in ("config_D", "x3d_T5"):
    m0 = pooled(a, 0, exclude=("2men_clash",)); m2 = pooled(a, 2, exclude=("2men_clash",))
    out[a] = {"home_tol0": m0, "home_tol2": m2}
    print(f"  {a:<10} {'home (17 scenes)':<22} | {m0['prec']:>6.1%} {m0['rec']:>6.1%} {m0['f1']:>6.1%} | "
          f"{m2['prec']:>6.1%} {m2['rec']:>6.1%} {m2['f1']:>6.1%} {m2['far']:>5.1%} | "
          f"{REF[a]['f1t2']:.1%}/{REF[a]['far']:.1%}")
    if "2men_clash" in scenes:
        labs = scenes["2men_clash"]["labels"]
        cm0 = binary_confusion_matrix(labs, scenes["2men_clash"][a])
        out[a]["clash_tol0_f1"] = cm0.f1
        print(f"  {a:<10} {'2men_clash (held-out)':<22} |   -      -    {cm0.f1:>6.1%} |")
(_ROOT / "reports" / "home_sessionbg_benchmark.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote reports/home_sessionbg_benchmark.json")
