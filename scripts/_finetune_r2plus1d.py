"""Fine-tune Kinetics-400-pretrained R(2+1)D-18 for contact detection.

Guy's spatio-temporal idea via transfer learning instead of scratch training:
  - 3 cameras -> the 3 "RGB" input channels
  - Tateno residual frames, z-scored then rescaled to Kinetics channel stats
  - 62x80 -> 112x112 bilinear upsample, T-frame sliding window (default 5)
  - stem + layer1 frozen; layer2..4 + new 2-class head fine-tuned
Same protocol as the scratch runs: per-session timeline split 60/15/25,
class-weighted CE, threshold swept on val, honest test + pooled full-replay.

Usage: python scripts/_finetune_r2plus1d.py [T]
Outputs reports/r2plus1d_T<T>_results.json + checkpoints/r2plus1d_contact/T<T>.pt
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

import eval_waveshare_contact as geo
from thermal_algorithms.preprocessing import TatenoPipeline
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

T = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TRAIN_FRAC, VAL_FRAC = 0.60, 0.15
EPOCHS, BATCH, LR, WD = 8, 8, 1e-4, 1e-4
SIZE = 112
KIN_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KIN_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH_GRID = [round(0.05 * k, 2) for k in range(1, 20)]
CKPT = _ROOT / "checkpoints" / "r2plus1d_contact" / f"T{T}.pt"
OUT_JSON = _ROOT / "reports" / f"r2plus1d_T{T}_results.json"

# ---------------------------------------------------------------------------
idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)

print(f"building per-session residual volumes (T={T}) ...", flush=True)
sessions = {}   # scene -> {"vol": (n,3,H,W) float32, "labels": [..], "splits": [..]}
for scene in sorted(labels):
    try: s = idx.find(scene)
    except Exception: continue
    if not all(ch in s.channels_with_data for ch in geo.CHANNELS): continue
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    if len(fis) < T + 2: continue
    n_tr = int(len(fis) * TRAIN_FRAC); n_va = int(len(fis) * (TRAIN_FRAC + VAL_FRAC))
    frames, labs, splits = [], [], []
    for k, fi in enumerate(fis):
        resids = [pre[ch].predict(geo.get_frame(s, ch, fi)).data.astype(np.float32)
                  for ch in geo.CHANNELS]
        frames.append(np.stack(resids, 0))
        labs.append(int(labels[scene][fi]))
        splits.append("train" if k < n_tr else ("val" if k < n_va else "test"))
    sessions[scene] = {"vol": np.stack(frames, 0), "labels": labs, "splits": splits}
print(f"  {len(sessions)} scenes, "
      f"{sum(v['vol'].shape[0] for v in sessions.values())} frames", flush=True)

# global z-score stats from TRAIN frames
train_px = np.concatenate([v["vol"][[i for i, sp in enumerate(v["splits"]) if sp == "train"]].ravel()
                           for v in sessions.values()])
G_MEAN, G_STD = float(train_px.mean()), float(train_px.std()) + 1e-6
print(f"  residual stats: mean={G_MEAN:.3f} std={G_STD:.3f}")

def make_clip(scene, start):
    """(3, T, SIZE, SIZE) tensor mapped onto Kinetics channel statistics."""
    v = sessions[scene]["vol"][start:start + T]              # (T, 3, H, W)
    x = torch.from_numpy(v).permute(1, 0, 2, 3)              # (3, T, H, W)
    x = (x - G_MEAN) / G_STD                                 # z-score
    x = x * KIN_STD + KIN_MEAN                               # match Kinetics stats
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    return x

# window specs over contiguous train runs (label = last frame, as in X3D runs)
specs = []
for scene, v in sessions.items():
    tr = [i for i, sp in enumerate(v["splits"]) if sp == "train"]
    if len(tr) < T: continue
    for start in range(tr[0], tr[-1] - T + 2):
        specs.append((scene, start, v["labels"][start + T - 1]))
pos = sum(s[2] for s in specs)
print(f"  train windows: {len(specs)} (pos={pos})")

# ---------------------------------------------------------------------------
model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
model.fc = nn.Linear(512, 2)
for name, p in model.named_parameters():
    if name.startswith(("stem", "layer1")):
        p.requires_grad = False
model = model.to(DEVICE)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  r2plus1d_18 on {DEVICE}; trainable params: {n_train/1e6:.1f}M")

neg = len(specs) - pos
weights = torch.tensor([len(specs) / (2 * max(neg, 1)), len(specs) / (2 * max(pos, 1))],
                       dtype=torch.float32).to(DEVICE)
crit = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WD)
rng = np.random.default_rng(0)

model.train()
t0 = time.time()
for epoch in range(EPOCHS):
    order = rng.permutation(len(specs))
    tot = 0.0; nb = 0
    for b in range(0, len(specs), BATCH):
        batch = [specs[i] for i in order[b:b + BATCH]]
        x = torch.stack([make_clip(sc, st) for sc, st, _ in batch]).to(DEVICE)
        y = torch.tensor([lb for _, _, lb in batch], dtype=torch.long, device=DEVICE)
        loss = crit(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item()); nb += 1
    print(f"    epoch {epoch+1:>2}/{EPOCHS}  loss={tot/max(1,nb):.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"T": T, "g_mean": G_MEAN, "g_std": G_STD,
            "state": model.state_dict()}, CKPT)
print(f"  saved {CKPT}")

# ---------------------------------------------------------------------------
print("scoring (sliding window over every session) ...", flush=True)
model.eval()
scored = []      # (split, scene, label, conf) — conf 0 while buffer fills
scene_confs = {}
with torch.no_grad():
    for scene, v in sessions.items():
        n = v["vol"].shape[0]
        confs = []
        for i in range(n):
            if i < T - 1:
                confs.append(0.0)
            else:
                x = make_clip(scene, i - T + 1).unsqueeze(0).to(DEVICE)
                confs.append(float(torch.softmax(model(x), 1)[0, 1]))
            scored.append((v["splits"][i], scene, v["labels"][i], confs[-1]))
        scene_confs[scene] = confs

def sweep(split):
    rows = [(l, c) for sp, _, l, c in scored if sp == split]
    best = None
    for th in THRESH_GRID:
        cm = binary_confusion_matrix([l for l, _ in rows], [1 if c > th else 0 for _, c in rows])
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (th, cm)
    return best

best_th, _ = sweep("val")
test_rows = [(l, c) for sp, _, l, c in scored if sp == "test"]
cm_test = binary_confusion_matrix([l for l, _ in test_rows],
                                  [1 if c > best_th else 0 for _, c in test_rows])
print(f"\n  honest test (th={best_th}): P={cm_test.precision:.1%} R={cm_test.recall:.1%} "
      f"F1={cm_test.f1:.1%} FAR={cm_test.false_alarm_rate:.1%}")

def pooled(th_, tol):
    tp = fn = fp = tn = 0
    for scene, v in sessions.items():
        labs = v["labels"]; yp = [1 if c > th_ else 0 for c in scene_confs[scene]]
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
print(f"  pooled tol=0 : P={m0['prec']:.1%} R={m0['rec']:.1%} F1={m0['f1']:.1%} FAR={m0['far']:.1%}")
print(f"  pooled tol=±2: P={m2['prec']:.1%} R={m2['rec']:.1%} F1={m2['f1']:.1%} FAR={m2['far']:.1%}")
print("\n  per-scene F1 (tol=0), contact scenes:")
per_scene = {}
for sc in sorted(sessions):
    labs = sessions[sc]["labels"]
    if not any(labs): continue
    yp = [1 if c > best_th else 0 for c in scene_confs[sc]]
    cm = binary_confusion_matrix(labs, yp)
    per_scene[sc] = cm.f1
    print(f"    {sc:<24} pos={sum(labs):>3}  F1={cm.f1:.1%}  (TP={cm.tp} FN={cm.fn} FP={cm.fp})")

OUT_JSON.write_text(json.dumps({
    "T": T, "threshold": best_th, "epochs": EPOCHS,
    "train_windows": len(specs), "train_pos": pos,
    "honest_test": {"prec": cm_test.precision, "rec": cm_test.recall,
                    "f1": cm_test.f1, "far": cm_test.false_alarm_rate},
    "full_replay": {"tol0": m0, "tol2": m2}, "per_scene_f1": per_scene,
}, indent=1))
print(f"\nwrote {OUT_JSON}")
