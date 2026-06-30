"""config-D over the ENTIRE dataset, per-video confusion matrices.
Fixed params (no tuning): raw SSD + residual blob + two-body merge (k=1, tau2=8)
+ T1 morphology (l_open=2, l_close=7). NOTE: SSD trained on 60% of these frames,
so this is a whole-dataset diagnostic, not a held-out test."""
import json, sys
from collections import defaultdict
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np, time
import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

LO, LC = 2, 7
idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
pre = geo.fit_preprocessors(idx)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
print("running config-D over all labeled frames ...", flush=True)
scenes = {}
t0=time.time(); n=0
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if not fis: continue
    recs=[]
    for fi in fis:
        cams=[{"boxes":[d.bbox for d in ssd.predict(geo.get_frame(s,ch,fi))],
               "resid":pre[ch].predict(geo.get_frame(s,ch,fi)).data.astype(np.float32)} for ch in geo.CHANNELS]
        r={"fi":fi,"label":int(labels[scene][fi]),"cams":cams}
        r["base"]=ab.v9core_decision(r,"resid")
        recs.append(r); n+=1
        if n%300==0: print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)",flush=True)
    scenes[scene]=recs

print("\n"+"="*92)
print("  config-D — per-video confusion matrices (ENTIRE dataset; morph lo=2,lc=7)")
print("="*92)
hdr=f"  {'Scene':<22} {'N':>4} {'pos':>4} | {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4} | {'Prec':>6} {'Rec':>6} {'F1':>6} {'FAR':>6}"
print(hdr); print("  "+"-"*(len(hdr)-2))
agg=BinaryConfusionMatrix(0,0,0,0); rows={}
for sc in sorted(scenes):
    rs=scenes[sc]
    sm=v8.morph([r["base"] for r in rs],LO,LC)
    yt=[r["label"] for r in rs]; yp=sm
    cm=binary_confusion_matrix(yt,yp); agg=agg+cm
    rows[sc]={"n":cm.total,"pos":cm.tp+cm.fn,"tp":cm.tp,"fn":cm.fn,"fp":cm.fp,"tn":cm.tn,
              "prec":cm.precision,"rec":cm.recall,"f1":cm.f1,"far":cm.false_alarm_rate}
    rec = f"{cm.recall:>6.1%}" if (cm.tp+cm.fn)>0 else "   -  "
    f1  = f"{cm.f1:>6.1%}" if (cm.tp+cm.fn)>0 else "   -  "
    prec= f"{cm.precision:>6.1%}" if (cm.tp+cm.fp)>0 else "   -  "
    print(f"  {sc:<22} {cm.total:>4} {cm.tp+cm.fn:>4} | {cm.tp:>4} {cm.fn:>4} {cm.fp:>4} {cm.tn:>4} | {prec} {rec} {f1} {cm.false_alarm_rate:>6.1%}")
print("  "+"-"*(len(hdr)-2))
print(f"  {'AGGREGATE (all)':<22} {agg.total:>4} {agg.tp+agg.fn:>4} | {agg.tp:>4} {agg.fn:>4} {agg.fp:>4} {agg.tn:>4} | "
      f"{agg.precision:>6.1%} {agg.recall:>6.1%} {agg.f1:>6.1%} {agg.false_alarm_rate:>6.1%}")
json.dump({"morph":[LO,LC],"aggregate":{"tp":agg.tp,"fn":agg.fn,"fp":agg.fp,"tn":agg.tn,
          "prec":agg.precision,"rec":agg.recall,"f1":agg.f1,"far":agg.false_alarm_rate},
          "per_video":rows}, open(_ROOT/"reports"/"configd_full_dataset_results.json","w"), indent=2, default=float)
print("\n  JSON -> reports/configd_full_dataset_results.json")
