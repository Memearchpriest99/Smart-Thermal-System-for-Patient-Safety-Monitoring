"""MLX -> Thermo-X3D transfer experiment.

Tests whether pretraining Thermo-X3D on the low-res MLX90640 touch data (real
contact events, 24x32, upsampled to 62x80) and fine-tuning on Waveshare beats
training on Waveshare alone. Three conditions, same Waveshare val/test split
(test = 695 frames, 41 pos), threshold tuned on val:

  scratch   : X3D trained on Waveshare train windows only.
  transfer  : X3D pretrained on upsampled-MLX touch windows, then fine-tuned on
              the same Waveshare train windows.
  (MLX uses scene-level weak contact labels: contact scenes positive, others
   negative; both domains use a Tateno residual representation.)

Outputs reports/mlx_transfer_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from scipy.ndimage import zoom

import eval_waveshare_contact as geo
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984, MLX90640
from thermal_algorithms.core.types import Frame
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 16
TRAIN_FRAC, VAL_FRAC = 0.60, 0.15
WH, WW = WAVESHARE_26984.height, WAVESHARE_26984.width        # 62, 80
PRE_EPOCHS, FT_EPOCHS = 12, 15
BATCH = 8

MLX_ROOT = "datasets/mlx90640"
MLX_EMPTY = "emptyroomwithpcscreen"
MLX_CONTACT = {"2pplwithtouch", "2pplfight", "setup2_2_man_shove",
               "setup2_3_man_fight", "setup2_3_man_hug_walk",
               "threepplclose", "threepplsemiclose"}
MLX_NONCONTACT = {"setup2_1_man_fall", "setup2_1_man_running", "setup2_1_man_walk",
                  "setup2_2_men_walk", "setup2_3_man_walk", "personpresence",
                  "personrunning", "onepersonstandsit", "crosswalk ppl"}

OUT_JSON = _ROOT / "reports" / "mlx_transfer_results.json"


# ---------------------------------------------------------------------------
# Build per-scene upsampled residual arrays + window specs
# ---------------------------------------------------------------------------

def build_mlx(idx):
    pre = {}
    empty = idx.find(MLX_EMPTY)
    for ch in (0, 1, 2):
        calib = [empty.load_frame(ch, i) for i in range(empty.n_frames) if i % 2 == 0]
        pre[ch] = TatenoPipeline(MLX90640).fit(calib)
    sess, specs = {}, []
    for scene in sorted(MLX_CONTACT | MLX_NONCONTACT):
        try:
            s = idx.find(scene)
        except Exception:
            continue
        if not all(ch in s.channels_with_data for ch in (0, 1, 2)):
            continue
        n = min(s.load_frames(ch).shape[0] for ch in (0, 1, 2))
        if n < T:
            continue
        arr = []
        for ch in (0, 1, 2):
            frames = s.load_frames(ch)[:n]
            res = np.stack([pre[ch].predict(Frame(data=frames[i], timestamp=0, camera_id=ch)).data
                            for i in range(n)], 0)
            up = zoom(res, (1, WH / res.shape[1], WW / res.shape[2]), order=1).astype(np.float32)
            arr.append(up)
        label = 1 if scene in MLX_CONTACT else 0
        sess[scene] = arr
        for start in range(n - T + 1):
            specs.append((scene, start, label))
    return sess, specs


def build_wave(idx):
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    sess, specs, order = {}, [], {}
    for scene in sorted(labels):
        try:
            s = idx.find(scene)
        except Exception:
            continue
        if not all(ch in s.channels_with_data for ch in (0, 1, 2)):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
        if len(fis) < T + 2:
            continue
        n_tr = int(len(fis) * TRAIN_FRAC); n_va = int(len(fis) * (TRAIN_FRAC + VAL_FRAC))
        arr = [np.stack([pre[ch].predict(geo.get_frame(s, ch, fi)).data for fi in fis], 0).astype(np.float32)
               for ch in (0, 1, 2)]
        lab = [int(labels[scene][fi]) for fi in fis]
        spl = ["train" if k < n_tr else ("val" if k < n_va else "test") for k in range(len(fis))]
        sess[scene] = arr
        order[scene] = (lab, spl)
        for start in range(len(fis) - T + 1):
            if spl[start + T - 1] == "train":
                specs.append((scene, start, lab[start + T - 1]))
    return sess, specs, order


# ---------------------------------------------------------------------------
# Manual X3D training over (scene,start,label) specs
# ---------------------------------------------------------------------------

def set_stats(det, sess, specs):
    px = np.concatenate([sess[sc][ch][st:st + T].ravel()
                         for sc, st, _ in specs[::5] for ch in range(3)])
    det._global_mean = float(px.mean()); det._global_std = float(px.std()) + 1e-6


def vol(det, sess, sc, st):
    a = sess[sc]
    return np.stack([np.stack([det._normalise(a[ch][st + t]) for t in range(T)], 0)
                     for ch in range(3)], 0)


def train(det, sess, specs, epochs, tag):
    model, dev = det._get_model()
    pos = sum(s[2] for s in specs); neg = len(specs) - pos
    w = torch.tensor([len(specs) / (2 * max(neg, 1)), len(specs) / (2 * max(pos, 1))],
                     dtype=torch.float32, device=dev)
    crit = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=det._lr, weight_decay=det._wd)
    rng = np.random.default_rng(0)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        ordr = rng.permutation(len(specs)); tot = 0.0; nb = 0
        for b in range(0, len(specs), BATCH):
            batch = [specs[i] for i in ordr[b:b + BATCH]]
            x = torch.from_numpy(np.stack([vol(det, sess, sc, st) for sc, st, _ in batch], 0)).to(dev)
            y = torch.tensor([l for _, _, l in batch], dtype=torch.long, device=dev)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print(f"      {tag} epoch {ep+1:>2}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    det._is_fitted = True
    print(f"      {tag} trained ({time.time()-t0:.0f}s)")


def evaluate(det, wsess, worder):
    """Replay each Waveshare scene in order; collect val/test (split, label, conf)."""
    scored = []
    for scene, (lab, spl) in worder.items():
        a = wsess[scene]; n = a[0].shape[0]
        det.reset()
        for i in range(n):
            triplet = tuple(Frame(data=a[ch][i], timestamp=0.0, camera_id=ch) for ch in range(3))
            ev = det.predict(triplet)
            if spl[i] in ("val", "test"):
                scored.append((spl[i], lab[i], float(ev.confidence)))
    return scored


def report(name, scored):
    val = [(l, c) for s, l, c in scored if s == "val"]
    best = None
    for thr in np.linspace(0.2, 0.95, 16):
        cm = binary_confusion_matrix([l for l, _ in val], [1 if c > thr else 0 for _, c in val])
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (thr, cm)
    thr = best[0]
    te = [(l, c) for s, l, c in scored if s == "test"]
    cm = binary_confusion_matrix([l for l, _ in te], [1 if c > thr else 0 for _, c in te])
    print(f"  {name:<22} thr={thr:.2f}  P={cm.precision:.1%} R={cm.recall:.1%} "
          f"F1={cm.f1:.1%} FAR={cm.false_alarm_rate:.1%}  (TP{cm.tp} FP{cm.fp} FN{cm.fn})")
    return {"thr": float(thr), "prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
            "far": cm.false_alarm_rate, "tp": cm.tp, "fp": cm.fp, "fn": cm.fn}


def main():
    print("=" * 78)
    print("  MLX -> Thermo-X3D TRANSFER EXPERIMENT")
    print("=" * 78)
    widx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=WAVESHARE_26984)
    midx = DatasetIndex(MLX_ROOT, sensor_profile=MLX90640)

    print("  [1] build MLX upsampled windows ...", flush=True)
    msess, mspecs = build_mlx(midx)
    print(f"      MLX windows: {len(mspecs)} (pos={sum(s[2] for s in mspecs)}) from {len(msess)} scenes")
    print("  [2] build Waveshare windows ...", flush=True)
    wsess, wspecs, worder = build_wave(widx)
    print(f"      Waveshare train windows: {len(wspecs)} (pos={sum(s[2] for s in wspecs)})")

    results = {}

    print("\n  [3] SCRATCH (Waveshare only) ...", flush=True)
    det_s = ThermoX3DDetector(WAVESHARE_26984, T=T, persistence_frames=1, device=DEVICE)
    set_stats(det_s, wsess, wspecs)
    train(det_s, wsess, wspecs, FT_EPOCHS, "scratch")
    sc_s = evaluate(det_s, wsess, worder)

    print("\n  [4] TRANSFER: pretrain on MLX, fine-tune on Waveshare ...", flush=True)
    det_t = ThermoX3DDetector(WAVESHARE_26984, T=T, persistence_frames=1, device=DEVICE)
    set_stats(det_t, msess, mspecs)
    train(det_t, msess, mspecs, PRE_EPOCHS, "MLX-pretrain")
    set_stats(det_t, wsess, wspecs)                  # reset normalisation to Waveshare
    train(det_t, wsess, wspecs, FT_EPOCHS, "WS-finetune")
    sc_t = evaluate(det_t, wsess, worder)

    print("\n" + "=" * 78)
    print("  RESULTS (Waveshare test: 41 contact / 654 no-contact)")
    print("=" * 78)
    print("  Reference: scratch X3D (V-series) F1 41.7%; config-D rule single-split 60.8%, CV ~44%")
    results["scratch"] = report("X3D scratch (WS)", sc_s)
    results["transfer"] = report("X3D MLX->WS transfer", sc_t)
    d = (results["transfer"]["f1"] - results["scratch"]["f1"]) * 100
    print(f"\n  Transfer effect: ΔF1 = {d:+.1f} pt  "
          f"({'MLX pretraining helps' if d > 0 else 'MLX pretraining does not help'})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
