"""Per-scene confusion + FN list for config-D (raw SSD + residual blob + 2body + T1 morph)."""
import sys
from collections import defaultdict
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
pre = geo.fit_preprocessors(idx)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
print("building config-D cache (raw SSD + residual) ...", flush=True)
recs = []
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if len(fis) < vv.T + 2: continue
    n_tr=int(len(fis)*vv.TRAIN_FRAC); n_va=int(len(fis)*(vv.TRAIN_FRAC+vv.VAL_FRAC))
    for k,fi in enumerate(fis):
        split="train" if k<n_tr else ("val" if k<n_va else "test")
        cams=[]
        for ch in geo.CHANNELS:
            rawF=geo.get_frame(s,ch,fi); resid=pre[ch].predict(rawF).data.astype(np.float32)
            cams.append({"boxes":[d.bbox for d in ssd.predict(rawF)],"raw":rawF.data.astype(np.float32),"resid":resid})
        recs.append({"split":split,"scene":scene,"fi":fi,"label":int(labels[scene][fi]),"cams":cams})
for r in recs: r["base"]=ab.v9core_decision(r,"resid")
scenes=defaultdict(list)
for r in recs: scenes[r["scene"]].append(r)
for sc in scenes: scenes[sc].sort(key=lambda r:r["fi"])
# tune morph (lo,lc) on val
def cmval(lo,lc):
    yt,yp=[],[]
    for sc,rs in scenes.items():
        sm=v8.morph([r["base"] for r in rs],lo,lc)
        for r,v in zip(rs,sm):
            if r["split"]=="val": yt.append(r["label"]); yp.append(v)
    return binary_confusion_matrix(yt,yp)
best=None
for lo in range(0,5):
    for lc in range(0,8):
        cm=cmval(lo,lc)
        if best is None or (cm.f1,cm.recall)>(best[1].f1,best[1].recall): best=((lo,lc),cm)
lo,lc=best[0]
pred={}
for sc,rs in scenes.items():
    sm=v8.morph([r["base"] for r in rs],lo,lc)
    for r,v in zip(rs,sm): pred[id(r)]=v
# test per-scene
by=defaultdict(lambda:([],[],[]))
for r in recs:
    if r["split"]=="test":
        by[r["scene"]][0].append(r["label"]); by[r["scene"]][1].append(pred[id(r)]); by[r["scene"]][2].append(r["fi"])
yt=[r["label"] for r in recs if r["split"]=="test"]; yp=[pred[id(r)] for r in recs if r["split"]=="test"]
agg=binary_confusion_matrix(yt,yp)
print(f"\nconfig-D test (morph lo={lo},lc={lc}): P={agg.precision:.1%} R={agg.recall:.1%} F1={agg.f1:.1%} FAR={agg.false_alarm_rate:.1%} TP={agg.tp} FN={agg.fn} FP={agg.fp}")
print("\nPer-scene (contact-bearing + any FP):")
for sc in sorted(by):
    yl,yq,fis=by[sc]; cm=binary_confusion_matrix(yl,yq); pos=cm.tp+cm.fn
    if pos>0 or cm.fp>0:
        fn_fis=[f for f,l,p in zip(fis,yl,yq) if l==1 and p==0]
        print(f"  {sc:<22} pos={pos:<3} TP={cm.tp:<3} FN={cm.fn:<3} FP={cm.fp:<3}  missed_frames={fn_fis}")
