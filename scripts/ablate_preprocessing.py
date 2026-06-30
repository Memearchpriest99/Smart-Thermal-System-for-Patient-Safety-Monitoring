"""Preprocessing ablation — does the Tateno pipeline actually help contact results?

The Tateno preprocessing (Gaussian smooth -> background subtract -> L1 residual)
feeds the contact pipeline in two places: (1) the SSD person detector is trained
and run on the residual; (2) the blob-merge test segments warm components on the
residual. This ablation isolates both, using a fixed detector (V9 core: two-body
blob-merge quorum + T1 morphology, NO Thermo-X3D, so only the box+blob path is
compared) on the matched 695-frame test set:

  A  Tateno SSD + residual blob   (full Tateno pipeline)
  B  Tateno SSD + raw-temp blob   (preprocessing only for the detector)
  C  raw SSD    + raw-temp blob   (no preprocessing at all)

A vs B  -> does background subtraction help the blob segmentation?
A vs C  -> does the whole preprocessing pipeline help end-to-end?

Requires retraining the SSD on raw frames (cached at a separate checkpoint).
Outputs reports/preprocessing_ablation_results.json.
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

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v6 as v6
import eval_waveshare_contact_v8 as v8
import eval_waveshare_contact_v9 as v9
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.label_io import load_yolo_labels, PERSON_CLASS_ID
from thermal_algorithms.training.metrics import binary_confusion_matrix

PROFILE = geo.PROFILE
CHANNELS = geo.CHANNELS
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RAW_CKPT = _ROOT / "checkpoints" / "mobilenet_ssd_detector" / "Waveshare_26984_raw.thalg"
OUT_JSON = _ROOT / "reports" / "preprocessing_ablation_results.json"


# ---------------------------------------------------------------------------
# Raw SSD training (mirrors geo.build_ssd_examples but on raw frames)
# ---------------------------------------------------------------------------

def build_ssd_examples_raw(idx):
    eh, ew = PROFILE.height, PROFILE.width
    empty = idx.find(geo.EMPTY_SCENE)
    ex = []
    for session in idx.sessions:
        if session.scene in (geo.EMPTY_SCENE, geo.CALIB_SCENE):
            continue
        for ch in CHANNELS:
            if ch not in session.channels_with_labels:
                continue
            labeled = sorted(int(p.stem.split("_", 1)[1])
                             for p in session.frames_dir(ch).glob("frame_*.txt"))
            if not labeled:
                continue
            tr, _, _ = geo._split_indices(labeled)
            for fi in tr:
                f = geo.get_frame(session, ch, fi)
                dets = load_yolo_labels(session.frames_dir(ch) / f"frame_{fi:05d}.txt",
                                        frame_shape=(eh, ew), camera_id=ch,
                                        class_filter=[PERSON_CLASS_ID])
                ex.append((f, dets))
    for ch in CHANNELS:
        odd = [i for i in range(empty.n_frames) if i % 2 == 1]
        tr, _, _ = geo._split_indices(odd)
        for fi in tr:
            ex.append((geo.get_frame(empty, ch, fi), []))
    return ex


def get_raw_ssd(idx):
    if RAW_CKPT.is_file():
        print(f"  Loaded raw SSD checkpoint → {RAW_CKPT}")
        return MobileNetSSDDetector.load(RAW_CKPT)
    print("  Training SSD on RAW frames ...", flush=True)
    ex = build_ssd_examples_raw(idx)
    print(f"    {len(ex)} examples (pos={sum(1 for _,d in ex if d)})")
    t = time.time()
    ssd = MobileNetSSDDetector(PROFILE, n_epochs=25, batch_size=16, learning_rate=1e-3, device=DEVICE)
    ssd.fit(ex, verbose=True)
    ssd.save(RAW_CKPT)
    print(f"    trained ({time.time()-t:.0f}s) → {RAW_CKPT}")
    return ssd


# ---------------------------------------------------------------------------
# Raw cache (SSD-on-raw boxes; raw stored in both 'raw' and 'resid' slots so the
# blob test segments on raw temperature)
# ---------------------------------------------------------------------------

def build_cache_raw(idx, ssd_raw, labels):
    recs = []
    for scene in sorted(labels):
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if len(fis) < vv.T + 2:
            continue
        n_tr = int(len(fis) * vv.TRAIN_FRAC); n_va = int(len(fis) * (vv.TRAIN_FRAC + vv.VAL_FRAC))
        for k, fi in enumerate(fis):
            split = "train" if k < n_tr else ("val" if k < n_va else "test")
            cams = []
            for ch in CHANNELS:
                rawF = geo.get_frame(session, ch, fi)
                dets = ssd_raw.predict(rawF)
                cams.append({"boxes": [d.bbox for d in dets],
                             "raw": rawF.data.astype(np.float32),
                             "resid": rawF.data.astype(np.float32)})   # blob on raw temp
            recs.append({"split": split, "scene": scene, "fi": fi,
                         "label": int(labels[scene][fi]), "cams": cams})
    return recs


# ---------------------------------------------------------------------------
# V9 core decision (no X3D) with selectable blob source
# ---------------------------------------------------------------------------

def v9core_decision(rec, blob_key):
    cleaned = [vv.merge_oversegmented([bb for bb in c["boxes"]]) for c in rec["cams"]]
    people = max((len(b) for b in cleaned), default=0)
    if people <= 1:
        return 0
    states = [v9._cam_state_2body(cleaned[c], rec["cams"][c][blob_key], v9.K, v9.TAU2)
              for c in range(3)]
    votes = states.count("T"); merged = states.count("M"); clear = "C" in states
    return 1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0


def eval_config(recs, blob_key):
    """V9 core + T1 morphology (tuned on val); return best test confusion."""
    for r in recs:
        r["_d"] = v9core_decision(r, blob_key)
    scenes = defaultdict(list)
    for r in recs:
        scenes[r["scene"]].append(r)
    for s in scenes:
        scenes[s].sort(key=lambda r: r["fi"])

    def cm_split(pred, split):
        yt = [r["label"] for r in recs if r["split"] == split]
        yp = [pred[id(r)] for r in recs if r["split"] == split]
        return binary_confusion_matrix(yt, yp)

    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            pred = {}
            for s, rs in scenes.items():
                sm = v8.morph([r["_d"] for r in rs], lo, lc)
                for r, v in zip(rs, sm):
                    pred[id(r)] = v
            cmv = cm_split(pred, "val")
            if best is None or (cmv.f1, cmv.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cmv, pred)
    return best[0], cm_split(best[2], "test")


def main():
    print("=" * 80)
    print("  PREPROCESSING ABLATION — does Tateno help contact results?")
    print("=" * 80)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=PROFILE)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    ssd_tat = MobileNetSSDDetector.load(geo.SSD_CKPT)
    pre = geo.fit_preprocessors(idx)
    print("  [1] Tateno cache (SSD-on-residual) ...", flush=True)
    tat_cache = v6.build_cache(idx, ssd_tat, pre, labels)

    ssd_raw = get_raw_ssd(idx)
    print("  [2] Raw cache (SSD-on-raw) ...", flush=True)
    raw_cache = build_cache_raw(idx, ssd_raw, labels)

    configs = [
        ("A  Tateno SSD + residual blob", tat_cache, "resid"),
        ("B  Tateno SSD + raw-temp blob", tat_cache, "raw"),
        ("C  raw SSD + raw-temp blob",    raw_cache, "resid"),  # raw stored in both slots
    ]
    rows = []
    for name, cache, blob_key in configs:
        params, cm = eval_config(cache, blob_key)
        rows.append((name, params, cm))

    print("\n" + "=" * 80)
    print("  RESULTS (V9 core, no X3D; test 41 contact / 654 no-contact)")
    print("=" * 80)
    hdr = f"  {'Config':<32} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FAR':>7}  morph"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name, params, cm in rows:
        print(f"  {name:<32} {cm.precision:>6.1%} {cm.recall:>7.1%} {cm.f1:>7.1%} "
              f"{cm.false_alarm_rate:>7.1%}  (lo={params[0]},lc={params[1]})")

    a, b, c = rows[0][2], rows[1][2], rows[2][2]
    print("\n  Interpretation:")
    print(f"    Residual vs raw blob (A vs B):  F1 {a.f1:.1%} vs {b.f1:.1%}  "
          f"(Δ {100*(a.f1-b.f1):+.1f}pt) — effect of bg-subtraction on blob seg")
    print(f"    Full Tateno vs full raw (A vs C): F1 {a.f1:.1%} vs {c.f1:.1%}  "
          f"(Δ {100*(a.f1-c.f1):+.1f}pt) — effect of preprocessing end-to-end")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {name: {"params": {"l_open": p[0], "l_close": p[1]},
                "test": {"prec": cm.precision, "rec": cm.recall, "f1": cm.f1,
                         "far": cm.false_alarm_rate, "tp": cm.tp, "tn": cm.tn,
                         "fp": cm.fp, "fn": cm.fn}}
         for name, p, cm in rows}, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
