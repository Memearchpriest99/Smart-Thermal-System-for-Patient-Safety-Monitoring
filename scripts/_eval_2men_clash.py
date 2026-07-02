"""Evaluate ALL contact detectors on 2men_clash — the first truly HELD-OUT
contact scene (human-labelled, never seen by any model, no consensus labels).

Scores: config-D rule, all four X3D checkpoints, and the r2plus1d fine-tune,
each at its previously-tuned threshold. tol=0 and +-2.
Outputs reports/eval_2men_clash.json.
"""
import json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import torch
import torch.nn.functional as F

import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

SCENE = "2men_clash"
LO, LC = 2, 7
X3DS = {
    "x3d_old":     ("Waveshare_26984.thalg", 0.75),
    "x3d_clean":   ("Waveshare_26984_clean.thalg", 0.1),
    "x3d_relabel": ("Waveshare_26984_relabel.thalg", 0.05),
    "x3d_T5":      ("Waveshare_26984_relabel_T5.thalg", 0.2),
}
R2P1D_CKPT = _ROOT / "checkpoints" / "r2plus1d_contact" / "T5.pt"
R2P1D_TH = 0.05

SESSION_BG = "--session-bg" in sys.argv

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
s = idx.find(SCENE)
fis = sorted(fi for fi in labels[SCENE] if 0 <= fi < s.n_frames)
labs = [int(labels[SCENE][fi]) for fi in fis]
print(f"{SCENE}: {len(fis)} frames, {sum(labs)} positive")

if SESSION_BG:
    # Session-local automatic background: per-pixel temporal MEDIAN of the
    # session itself (robust to moving people; no per-room calibration).
    from thermal_algorithms.preprocessing import TatenoPipeline
    pre = {}
    for ch in geo.CHANNELS:
        stack = np.stack([geo.get_frame(s, ch, fi).data for fi in fis], 0)
        med = np.median(stack, axis=0)
        from thermal_algorithms.core.types import Frame
        pre[ch] = TatenoPipeline(geo.PROFILE).fit([Frame(data=med, timestamp=0.0, camera_id=ch)])
    print("using SESSION-LOCAL median background")
else:
    pre = geo.fit_preprocessors(idx)
    print("using empty_room-fit background (old room)")

# shared per-frame data
resid_seq, dets_seq = [], []
for fi in fis:
    raws = [geo.get_frame(s, ch, fi) for ch in geo.CHANNELS]
    resids = tuple(pre[ch].predict(raws[c]) for c, ch in enumerate(geo.CHANNELS))
    resid_seq.append(resids)
    dets_seq.append([ssd_raw.predict(rw) for rw in raws])

preds = {}

# config-D
base = [ab.v9core_decision(
    {"cams": [{"boxes": [d.bbox for d in dets_seq[i][c]],
               "resid": resid_seq[i][c].data.astype(np.float32)} for c in range(3)]}, "resid")
    for i in range(len(fis))]
preds["config_D"] = list(v8.morph(base, LO, LC))

# X3D variants
for name, (ck, th) in X3DS.items():
    det = ThermoX3DDetector.load(_ROOT / "checkpoints" / "thermo_x3d_detector" / ck)
    det.reset()
    preds[name] = [1 if float(det.predict(r).confidence) > th else 0 for r in resid_seq]

# r2plus1d fine-tune
from torchvision.models.video import r2plus1d_18
ck = torch.load(R2P1D_CKPT, map_location="cpu", weights_only=False)
T, G_MEAN, G_STD = ck["T"], ck["g_mean"], ck["g_std"]
model = r2plus1d_18()
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(ck["state"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(DEVICE).eval()
KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
vol = np.stack([np.stack([r[c].data.astype(np.float32) for c in range(3)], 0)
                for r in resid_seq], 0)          # (n, 3, H, W)
confs = []
with torch.no_grad():
    for i in range(len(fis)):
        if i < T - 1:
            confs.append(0.0); continue
        x = torch.from_numpy(vol[i - T + 1:i + 1]).permute(1, 0, 2, 3)
        x = (x - G_MEAN) / G_STD * KIN_STD + KIN_MEAN
        x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
        confs.append(float(torch.softmax(model(x.unsqueeze(0).to(DEVICE)), 1)[0, 1]))
preds["r2plus1d_T5"] = [1 if c > R2P1D_TH else 0 for c in confs]

# ---- score -------------------------------------------------------------------
def score(yp, tol):
    tp = fn = fp = tn = 0
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

print("\n" + "=" * 86)
print(f"  HELD-OUT scene {SCENE} ({sum(labs)} pos / {len(labs)} frames)")
print(f"  {'model':<14} | {'P t0':>6} {'R t0':>6} {'F1 t0':>6} {'FAR':>5} | {'P ±2':>6} {'R ±2':>6} {'F1 ±2':>6} {'FAR':>5}")
print("  " + "-" * 82)
out = {}
for name, yp in preds.items():
    m0, m2 = score(yp, 0), score(yp, 2)
    out[name] = {"tol0": m0, "tol2": m2}
    print(f"  {name:<14} | {m0['prec']:>6.1%} {m0['rec']:>6.1%} {m0['f1']:>6.1%} {m0['far']:>5.1%} | "
          f"{m2['prec']:>6.1%} {m2['rec']:>6.1%} {m2['f1']:>6.1%} {m2['far']:>5.1%}")

(_ROOT / "reports" / "eval_2men_clash.json").write_text(json.dumps(
    {"scene": SCENE, "n_frames": len(labs), "n_pos": sum(labs), "results": out}, indent=1))
print(f"\nwrote reports/eval_2men_clash.json")
