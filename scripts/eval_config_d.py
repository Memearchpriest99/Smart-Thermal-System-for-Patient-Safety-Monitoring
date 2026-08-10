#!/usr/bin/env python3
"""Evaluate config-D (temporal two-body blob-merge contact rule) on the SAME
held-out contact test scenes every other contact detector is scored on.

Config-D is derived in full in reports/algorithm_derivations.md S8. In brief:
per-camera person boxes from a RAW-input MobileNet-SSD, over-segmented boxes
merged, a two-body blob-merge test on the Tateno RESIDUAL (a component of the
adaptively-thresholded warm mask containing exactly two box centres means the
two people have thermally merged, i.e. contact), a veto-based cross-camera
quorum, and finally 1-D morphological opening/closing on the per-frame
decision stream.

It is not a ThermalAlgorithm and has no checkpoint, so eval_all_detectors.py
cannot score it; this script produces a row in the same JSON shape so the
report can table it alongside the others.

PROTOCOL -- two deliberate departures from the historical config-D runs, both
making this number STRICTER and therefore comparable to ThermoX3D's:

1. The raw SSD is trained with the contact test scenes EXCLUDED. The original
   runs trained the person detector on a per-scene timeline split that
   included frames from the very scenes contact was then scored on -- fine for
   a person detector, but it leaks into a contact evaluation. Excluding them
   costs some detector quality and is the honest choice.
2. The morphology parameters (l_open, l_close) are tuned on NON-TEST scenes
   only, then applied unchanged to the test scenes. The historical 60.8%
   figure came from a within-scene timeline split; the honest cross-validated
   figure reported alongside it was 44.3% (see historical_investigations.md
   S4.1).

Outputs reports/config_d_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v8 as v8
import eval_waveshare_contact_v9 as v9
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.label_io import PERSON_CLASS_ID, load_yolo_labels
from thermal_algorithms.training.metrics import binary_confusion_matrix
from thermal_algorithms.training.split import CONTACT_TEST_SCENES

DATA_ROOT = _REPO_ROOT.parent / "data" / "waveshare_work"
RAW_CKPT = _REPO_ROOT / "checkpoints" / "mobilenet_ssd_detector" / "Waveshare_26984_rawD.thalg"
OUT_JSON = _REPO_ROOT / "reports" / "config_d_eval.json"
CHANNELS = geo.CHANNELS


def build_raw_ssd_examples(idx, exclude: set[str]):
    """Person-box examples on RAW frames, from every scene except the contact
    test scenes (see PROTOCOL note 1) and the calibration scene."""
    ex = []
    for session in idx.sessions:
        if session.scene in exclude or session.scene == geo.CALIB_SCENE:
            continue
        for ch in CHANNELS:
            if ch not in session.channels_with_labels:
                continue
            fdir = session.frames_dir(ch)
            labeled = sorted(int(p.stem.split("_", 1)[1]) for p in fdir.glob("frame_*.txt"))
            for fi in labeled:
                if not (0 <= fi < session.n_frames):
                    continue
                dets = load_yolo_labels(fdir / f"frame_{fi:05d}.txt",
                                        frame_shape=(WAVESHARE_26984.height, WAVESHARE_26984.width),
                                        camera_id=ch, class_filter=[PERSON_CLASS_ID])
                ex.append((geo.get_frame(session, ch, fi), dets))
    return ex


def get_raw_ssd(idx, exclude: set[str], epochs: int, device: str):
    if RAW_CKPT.is_file():
        print(f"  loaded raw SSD -> {RAW_CKPT.name}")
        return MobileNetSSDDetector.load(RAW_CKPT)
    print("  training MobileNet-SSD on RAW frames (test scenes excluded) ...", flush=True)
    ex = build_raw_ssd_examples(idx, exclude)
    print(f"    {len(ex)} examples ({sum(1 for _f, d in ex if d)} with boxes)", flush=True)
    t = time.time()
    ssd = MobileNetSSDDetector(WAVESHARE_26984, n_epochs=epochs, batch_size=16,
                               learning_rate=1e-3, device=device)
    ssd.fit(ex)
    RAW_CKPT.parent.mkdir(parents=True, exist_ok=True)
    ssd.save(RAW_CKPT)
    print(f"    trained in {time.time() - t:.0f}s -> {RAW_CKPT.name}")
    return ssd


def scene_decisions(session, frame_indices, ssd, pre):
    """Per-frame config-D decision (pre-morphology) for one scene."""
    out = []
    for fi in frame_indices:
        boxes, resids = [], []
        for ch in CHANNELS:
            raw = geo.get_frame(session, ch, fi)
            boxes.append([d.bbox for d in ssd.predict(raw)])          # RAW -> detector
            resids.append(pre[ch].predict(raw).data.astype(np.float32))  # RESIDUAL -> blob test
        cleaned = [vv.merge_oversegmented(b) for b in boxes]
        people = max((len(b) for b in cleaned), default=0)
        if people <= 1:
            out.append(0)
            continue
        states = [v9._cam_state_2body(cleaned[c], resids[c], v9.K, v9.TAU2) for c in range(3)]
        votes, merged = states.count("T"), states.count("M")
        clear = "C" in states
        out.append(1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--ssd-epochs", type=int, default=25)
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.data_root)
    idx = DatasetIndex(root, sensor_profile=WAVESHARE_26984)
    labels = geo._load_contact_labels(root / "contact_labels.csv")
    test_scenes = sorted(set(CONTACT_TEST_SCENES) & set(labels))
    tune_scenes = sorted(s for s in labels if s not in CONTACT_TEST_SCENES
                         and any(labels[s].values()))

    print("=" * 78)
    print("  Config-D -- temporal two-body blob-merge contact rule")
    print("=" * 78)
    print(f"  tune scenes (morphology only): {tune_scenes}")
    print(f"  test scenes (held out):        {test_scenes}")

    ssd = get_raw_ssd(idx, set(CONTACT_TEST_SCENES), args.ssd_epochs, device)
    print("  fitting Tateno preprocessors (residual path) ...", flush=True)
    pre = geo.fit_preprocessors(idx)

    per_scene = {}
    for scene in tune_scenes + test_scenes:
        try:
            session = idx.find(scene)
        except Exception:
            continue
        if not all(ch in session.channels_with_data for ch in CHANNELS):
            continue
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < session.n_frames)
        if not fis:
            continue
        t = time.time()
        per_scene[scene] = {
            "fis": fis,
            "y": [labels[scene][fi] for fi in fis],
            "d": scene_decisions(session, fis, ssd, pre),
        }
        print(f"    {scene:<24} {len(fis):>5} frames  ({time.time() - t:.0f}s)", flush=True)

    # ---- tune (l_open, l_close) on non-test scenes only -------------------
    best = None
    for lo in range(0, 7):
        for lc in range(0, 9):
            yt, yp = [], []
            for s in tune_scenes:
                if s not in per_scene:
                    continue
                yt += per_scene[s]["y"]
                yp += v8.morph(per_scene[s]["d"], lo, lc)
            if not yt:
                continue
            cm = binary_confusion_matrix(yt, yp)
            if best is None or (cm.f1, cm.recall) > (best[1].f1, best[1].recall):
                best = ((lo, lc), cm)
    (lo, lc), tune_cm = best
    print(f"\n  tuned morphology: l_open={lo}, l_close={lc} "
          f"(tune-set F1={tune_cm.f1:.3f})")

    # ---- apply unchanged to the held-out test scenes ----------------------
    rows, yt_all, yp_all = [], [], []
    for s in test_scenes:
        if s not in per_scene:
            continue
        y = per_scene[s]["y"]
        p = v8.morph(per_scene[s]["d"], lo, lc)
        cm = binary_confusion_matrix(y, p)
        rows.append({"scene": s, "n": len(y), "f1": cm.f1, "precision": cm.precision,
                     "recall": cm.recall, "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn})
        yt_all += y
        yp_all += p
    agg = binary_confusion_matrix(yt_all, yp_all)

    print("\n  === held-out test scenes ===")
    for r in rows:
        print(f"    {r['scene']:<22} n={r['n']:<5} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}")
    print(f"    {'AGGREGATE':<22} n={agg.total:<5} P={agg.precision:.3f} R={agg.recall:.3f} "
          f"F1={agg.f1:.3f}  FAR={agg.false_alarm_rate:.3f}")
    print(f"    tp={agg.tp} tn={agg.tn} fp={agg.fp} fn={agg.fn}")

    payload = {
        "checkpoints_dir": "(none -- rule-based pipeline)",
        "contact_test_scenes": test_scenes,
        "contact_tune_scenes": tune_scenes,
        "morphology": {"l_open": lo, "l_close": lc},
        "protocol_note": (
            "Raw SSD trained with contact test scenes EXCLUDED; morphology tuned on non-test "
            "scenes only. Stricter than the historical config-D protocol, so this number is "
            "directly comparable to the other contact detectors in this report."
        ),
        "per_scene": rows,
        "results": [{
            "detector_name": "RBTCT (Rule-Based Temporal Contact Tracker)",
            "variant": "fp32_baseline",
            "task": "contact",
            "n_frames": agg.total,
            "accuracy": agg.accuracy, "precision": agg.precision,
            "recall": agg.recall, "f1": agg.f1, "mean_iou": None,
            "mean_inference_ms": None, "p95_inference_ms": None,
            "tp": agg.tp, "tn": agg.tn, "fp": agg.fp, "fn": agg.fn,
        }],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
