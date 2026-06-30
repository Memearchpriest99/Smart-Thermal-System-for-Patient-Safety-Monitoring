"""Contact detection — deep temporal alternatives (§ 4.4.3.2 / § 4.4.3.3).

Trains and evaluates the two learned contact detectors on the Waveshare
dataset, alongside the geometric baseline already covered by
``eval_waveshare_contact.py``:

  * MV-STGCN  (§ 4.4.3.2) — Multi-View Spatiotemporal Graph CNN. Front-end is
    MobileNet-SSD + homographic fusion → actor graph; a sliding window of
    T=16 actor graphs feeds an adaptive-adjacency ST-GCN. Inherits the
    homography of the geometric path.
  * Thermo-X3D (§ 4.4.3.3) — pixel-based (2+1)D X3D over raw thermal volumes.
    No detections, no homography — the failsafe for when contact merges two
    people into one blob.

Shared setup (Tateno preprocessing, self-calibrated homography, the trained
SSD checkpoint) is reused from ``eval_waveshare_contact.py``.

Protocol
--------
* Temporal models need contiguous frames, so each session's timeline is split
  (no shuffle): first TRAIN_FRAC → train, next VAL_FRAC → val, rest → test.
* At eval the detector is reset per session and the whole session is replayed
  in order so the T-frame rolling buffer is warm by the time we score val/test
  frames (warm-up frames are not scored).
* Raw per-frame confidence is captured once; the decision threshold is then
  swept on val (max F1) and applied to test.
* Heavy class imbalance (~7% positive) is countered with class-weighted CE.

Caveat: only ~188 positive contact frames exist in the whole dataset. These
models are data-starved; treat the numbers as a feasibility signal, not a
benchmark.

Outputs JSON to reports/waveshare_contact_dl_results.json.
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

import eval_waveshare_contact as geo
from thermal_algorithms.core.types import ContactEvent
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix, BinaryConfusionMatrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T = 16
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15            # remainder (0.25) → test
EPSILON_PX = 15.0          # cam0-plane clustering radius (best from geometric sweep)
THRESH_GRID = [round(0.05 * k, 2) for k in range(1, 20)]   # 0.05 .. 0.95

X3D_EPOCHS = 20
X3D_BATCH = 8
STGCN_EPOCHS = 40
STGCN_BATCH = 16

OUT_JSON = _ROOT / "reports" / "waveshare_contact_dl_results.json"
X3D_CKPT = _ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984.thalg"
STGCN_CKPT = _ROOT / "checkpoints" / "mv_stgcn_detector" / "_default.thalg"


# ---------------------------------------------------------------------------
# Build per-session temporal sequences (preprocessed frames + SSD detections)
# ---------------------------------------------------------------------------

def build_sequences(idx, pre, ssd):
    """Return {scene: [record,...]} in frame order. Each record:
    {label, triplet (3 preprocessed Frames), dets (3 Detection lists), split}."""
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    sequences: dict[str, list] = {}
    t0 = time.time(); n = 0
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        frame_indices = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(frame_indices) < T + 2:
            continue
        n_tr = int(len(frame_indices) * TRAIN_FRAC)
        n_va = int(len(frame_indices) * (TRAIN_FRAC + VAL_FRAC))
        recs = []
        for k, fi in enumerate(frame_indices):
            triplet = tuple(pre[ch].predict(geo.get_frame(session, ch, fi)) for ch in CHANNELS)
            dets = tuple(ssd.predict(t) for t in triplet)
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            recs.append({"label": int(labels[scene][fi]), "triplet": triplet,
                         "dets": dets, "split": split})
            n += 1
            if n % 300 == 0:
                print(f"      ... {n} frames  ({(time.time()-t0)/n*1000:.0f} ms/frame)", flush=True)
        sequences[scene] = recs
    return sequences


def _event(label: int, actors=()):  # binary-label ContactEvent for training
    return ContactEvent(actors=tuple(actors), pairs_in_contact=((0, 1),) if label else (),
                        timestamp=0.0, confidence=float(label))


def _class_weights(labels: list[int]) -> "torch.Tensor":
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return torch.tensor([1.0, 1.0])
    # inverse-frequency weights
    return torch.tensor([len(labels) / (2 * neg), len(labels) / (2 * pos)], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Threshold sweep + reporting helpers
# ---------------------------------------------------------------------------

def sweep_threshold(scored):
    """scored: list of (split, scene, label, confidence). Tune on val, report test."""
    val = [(l, c) for s, _, l, c in scored if s == "val"]
    best = None
    sweep = []
    for th in THRESH_GRID:
        yt = [l for l, _ in val]
        yp = [1 if c > th else 0 for _, c in val]
        cm = binary_confusion_matrix(yt, yp)
        sweep.append((th, cm))
        if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
            best = (th, cm)
    best_th = best[0]

    by_scene = defaultdict(lambda: ([], []))
    for s, scene, l, c in scored:
        if s != "test":
            continue
        by_scene[scene][0].append(l)
        by_scene[scene][1].append(1 if c > best_th else 0)
    rows, agg = [], BinaryConfusionMatrix(0, 0, 0, 0)
    for scene in sorted(by_scene):
        yt, yp = by_scene[scene]
        cm = binary_confusion_matrix(yt, yp)
        agg = agg + cm
        rows.append({"scene": scene, "acc": cm.accuracy, "prec": cm.precision,
                     "rec": cm.recall, "f1": cm.f1, "far": cm.false_alarm_rate,
                     "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "total": cm.total})
    return best_th, sweep, rows, agg


def report(name, best_th, rows, agg):
    print("\n" + "=" * 78)
    print(f"  {name}   (decision threshold = {best_th})")
    print("=" * 78)
    geo._print_scene_table(rows, f"Per-scene [test]")
    geo._print_cm(agg, "Aggregate [test]")


# ===========================================================================
# Thermo-X3D
# ===========================================================================

def run_thermo_x3d(sequences):
    print("\n" + "#" * 78)
    print("#  THERMO-X3D  (pixel-based, no homography)")
    print("#" * 78, flush=True)
    det = ThermoX3DDetector(PROFILE, T=T, persistence_frames=1,
                            n_epochs=X3D_EPOCHS, batch_size=X3D_BATCH, device=DEVICE)

    # Per-session contiguous arrays for fast windowed sampling.
    sess_arr = {}   # scene -> (vols[3] each (n,H,W) float32, labels list, splits list)
    for scene, recs in sequences.items():
        arr = [np.stack([r["triplet"][ch].data.astype(np.float32) for r in recs], 0)
               for ch in range(3)]
        sess_arr[scene] = (arr, [r["label"] for r in recs], [r["split"] for r in recs])

    # Global normalisation stats from TRAIN frames only.
    train_px = [a[ch][[i for i, s in enumerate(sp) if s == "train"]]
                for (a, _, sp) in sess_arr.values() for ch in range(3)]
    train_px = np.concatenate([p.ravel() for p in train_px if p.size])
    det._global_mean = float(train_px.mean()); det._global_std = float(train_px.std()) + 1e-6
    print(f"  global mean={det._global_mean:.2f}  std={det._global_std:.2f}")

    # Window specs over contiguous TRAIN runs.
    specs = []   # (scene, start, label)
    for scene, (a, labels, sp) in sess_arr.items():
        train_idx = [i for i, s in enumerate(sp) if s == "train"]
        if len(train_idx) < T:
            continue
        lo, hi = train_idx[0], train_idx[-1]
        for start in range(lo, hi - T + 2):
            specs.append((scene, start, labels[start + T - 1]))
    print(f"  X3D train windows: {len(specs)}  (pos={sum(s[2] for s in specs)})")

    model, device = det._get_model()
    weights = _class_weights([s[2] for s in specs]).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=det._lr, weight_decay=det._wd)
    rng = np.random.default_rng(0)

    def build_vol(scene, start):
        a = sess_arr[scene][0]
        vol = np.stack([np.stack([det._normalise(a[ch][start + t]) for t in range(T)], 0)
                        for ch in range(3)], 0)   # (3, T, H, W)
        return vol

    model.train()
    t0 = time.time()
    for epoch in range(X3D_EPOCHS):
        order = rng.permutation(len(specs))
        tot = 0.0; nb = 0
        for b in range(0, len(specs), X3D_BATCH):
            batch = [specs[i] for i in order[b:b + X3D_BATCH]]
            vols = np.stack([build_vol(sc, st) for sc, st, _ in batch], 0)
            x = torch.from_numpy(vols).to(device)
            yb = torch.tensor([lb for _, _, lb in batch], dtype=torch.long, device=device)
            logits = model(x)
            loss = crit(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print(f"    epoch {epoch+1:>3}/{X3D_EPOCHS}  loss={tot/max(1,nb):.4f}", flush=True)
    det._is_fitted = True
    det.save(X3D_CKPT)
    print(f"  trained ({time.time()-t0:.1f}s) → {X3D_CKPT}")

    # Eval: replay each session in order; score val/test frames.
    scored = []
    for scene, recs in sequences.items():
        det.reset()
        for r in recs:
            ev = det.predict(r["triplet"])
            if r["split"] in ("val", "test"):
                scored.append((r["split"], scene, r["label"], ev.confidence))
    best_th, sweep, rows, agg = sweep_threshold(scored)
    report("THERMO-X3D", best_th, rows, agg)
    return {"best_threshold": best_th,
            "sweep_val": [{"th": th, "f1": cm.f1, "prec": cm.precision, "rec": cm.recall}
                          for th, cm in sweep],
            "test": {"aggregate": _agg_dict(agg), "per_scene": rows}}


# ===========================================================================
# MV-STGCN
# ===========================================================================

def run_mv_stgcn(sequences, H):
    print("\n" + "#" * 78)
    print("#  MV-STGCN  (SSD + homographic fusion + ST-GCN)")
    print("#" * 78, flush=True)
    det = MVSTGCNDetector(PROFILE, homography=H, epsilon_m=EPSILON_PX, T=T,
                          persistence_frames=1, n_epochs=STGCN_EPOCHS,
                          batch_size=STGCN_BATCH, device=DEVICE)

    # Build training windows per-session (respects boundaries), using SSD+fusion actors.
    windows = []
    for scene, recs in sequences.items():
        train_recs = [r for r in recs if r["split"] == "train"]
        if len(train_recs) < T:
            continue
        examples = []
        for r in train_recs:
            actors, _ = fuse_detections(list(r["dets"]), H, epsilon_m=EPSILON_PX, next_track_id=0)
            examples.append((r["triplet"], _event(r["label"], actors)))
        windows += det._build_training_windows(examples)
    pos = sum(int(w[3]) for w in windows)
    print(f"  STGCN train windows: {len(windows)}  (pos={pos})")

    model, device = det._get_model()
    weights = _class_weights([int(w[3]) for w in windows]).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=det._lr, weight_decay=det._wd)
    rng = np.random.default_rng(0)

    model.train()
    t0 = time.time()
    for epoch in range(STGCN_EPOCHS):
        order = rng.permutation(len(windows))
        tot = 0.0; nb = 0
        for b in range(0, len(windows), STGCN_BATCH):
            batch = [windows[i] for i in order[b:b + STGCN_BATCH]]
            fb, pb, mb, lb = det._collate(batch)
            logits = model(fb.to(device), pb.to(device), mb.to(device))
            loss = crit(logits, lb.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        print(f"    epoch {epoch+1:>3}/{STGCN_EPOCHS}  loss={tot/max(1,nb):.4f}", flush=True)
    det._is_fitted = True
    det.save(STGCN_CKPT)
    print(f"  trained ({time.time()-t0:.1f}s) → {STGCN_CKPT}")

    # Eval: replay each session; pass SSD detections so the front-end runs live.
    scored = []
    for scene, recs in sequences.items():
        det.reset()
        for r in recs:
            ev = det.predict(r["triplet"], detections=r["dets"])
            if r["split"] in ("val", "test"):
                scored.append((r["split"], scene, r["label"], ev.confidence))
    best_th, sweep, rows, agg = sweep_threshold(scored)
    report("MV-STGCN", best_th, rows, agg)
    return {"best_threshold": best_th,
            "sweep_val": [{"th": th, "f1": cm.f1, "prec": cm.precision, "rec": cm.recall}
                          for th, cm in sweep],
            "test": {"aggregate": _agg_dict(agg), "per_scene": rows}}


def _agg_dict(agg):
    return {"acc": agg.accuracy, "prec": agg.precision, "rec": agg.recall, "f1": agg.f1,
            "far": agg.false_alarm_rate, "tp": agg.tp, "tn": agg.tn, "fp": agg.fp,
            "fn": agg.fn, "total": agg.total}


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("  CONTACT DETECTION — Deep temporal models on Waveshare 26984")
    print("=" * 78)
    print(f"  Device: {DEVICE}   T={T}   split=train{TRAIN_FRAC}/val{VAL_FRAC}/test(rest)")

    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)

    if not geo.SSD_CKPT.is_file():
        print("  ERROR: SSD checkpoint missing — run eval_waveshare_contact.py first.")
        sys.exit(1)
    ssd = MobileNetSSDDetector.load(geo.SSD_CKPT)
    print(f"  Loaded SSD checkpoint → {geo.SSD_CKPT}")

    print("\n  Self-calibrating homography (for MV-STGCN front-end) ...", flush=True)
    H, _, n_track = geo.calibrate_homography(idx)
    print(f"    {n_track} track frames, ε={EPSILON_PX}px")

    print("\n  Calibrating Tateno + building per-session sequences ...", flush=True)
    pre = geo.fit_preprocessors(idx)
    sequences = build_sequences(idx, pre, ssd)
    nfr = sum(len(r) for r in sequences.values())
    npos = sum(rr["label"] for r in sequences.values() for rr in r)
    print(f"    {len(sequences)} sessions, {nfr} frames (pos={npos})")

    results = {"profile": PROFILE.name, "device": DEVICE, "T": T,
               "split": {"train": TRAIN_FRAC, "val": VAL_FRAC},
               "detectors": {}}
    results["detectors"]["ThermoX3D"] = run_thermo_x3d(sequences)
    results["detectors"]["MVSTGCN"] = run_mv_stgcn(sequences, H)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\n  JSON results → {OUT_JSON}")


if __name__ == "__main__":
    main()
