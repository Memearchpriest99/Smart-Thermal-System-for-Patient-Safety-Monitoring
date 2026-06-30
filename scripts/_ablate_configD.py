"""Config D — raw SSD boxes + Tateno residual blob (isolates detector preprocessing)."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
print("Building cache D (raw SSD boxes + residual blob) ...", flush=True)
recs = []
for scene in sorted(labels):
    try: session = idx.find(scene)
    except Exception: continue
    if not all(ch in session.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
    if len(fis) < vv.T + 2: continue
    n_tr = int(len(fis)*vv.TRAIN_FRAC); n_va = int(len(fis)*(vv.TRAIN_FRAC+vv.VAL_FRAC))
    for k, fi in enumerate(fis):
        split = "train" if k < n_tr else ("val" if k < n_va else "test")
        cams = []
        for ch in geo.CHANNELS:
            rawF = geo.get_frame(session, ch, fi)
            dets = ssd_raw.predict(rawF)                 # raw SSD boxes
            resid = pre[ch].predict(rawF).data           # Tateno residual for blob
            cams.append({"boxes":[d.bbox for d in dets], "raw":rawF.data.astype(np.float32),
                         "resid":resid.astype(np.float32)})
        recs.append({"split":split,"scene":scene,"fi":fi,"label":int(labels[scene][fi]),"cams":cams})
params, cm = ab.eval_config(recs, "resid")
print(f"\nD  raw SSD + residual blob:  P={cm.precision:.1%} R={cm.recall:.1%} "
      f"F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}  (lo={params[0]},lc={params[1]})")
print("  vs A (Tateno SSD + residual blob): F1 54.5%")
