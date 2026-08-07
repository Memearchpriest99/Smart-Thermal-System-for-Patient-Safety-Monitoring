#!/usr/bin/env python3
"""Write a JSON sidecar next to every checkpoint (.thalg) and every ONNX
export, describing everything needed to actually use the weight file:
algorithm/class, sensor profile, hyperparameters, what data trained it
(train/val/test split, synthetic vs real frame counts), input/output tensor
shapes, and a load snippet. Also folds in evaluated metrics/latency per
variant when available (from reports/full_corpus_eval_*.json /
reports/eval_*smoketest*.json), so the sidecar is self-contained -- someone
handed just the weight file + this JSON shouldn't need to read the report to
know how it performs.

Reads the checkpoint's raw pickle payload directly (format_version/
class_name/sensor_profile_name/params) rather than reconstructing a live
instance -- this only needs metadata, not a runnable model, and avoids
needing torch/CUDA available just to describe a checkpoint.

Usage::

    python scripts/generate_weight_metadata.py
    python scripts/generate_weight_metadata.py --checkpoints checkpoints_full_corpus
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

REPORTS_DIR = _REPO_ROOT / "reports"
ONNX_DIR = _REPO_ROOT.parent / "Weights" / "onnx"

# What each detector was actually trained on -- see scripts/train_full_corpus.py
# and data/DATASET_NOTES.md. Hand-written because it reflects a design
# decision (which sources feed which model), not something derivable purely
# from the checkpoint file itself.
TRAINING_DATA_NOTES = {
    "FireSVMDetector": (
        "waveshare_work TRAIN scenes (14/17, session-level seeded split, "
        "session_train_test_split default seed=0) + the FULL synth_room_1..5 "
        "corpus (~31M frame-instances, used wholesale -- synthetic data is never "
        "split, per the task-3 instruction). Synthetic labels are frame-level "
        "presence only (label_join.classify_event); waveshare labels are real "
        "per-frame YOLO class-0 boxes."
    ),
    "HOGSVMDetector": (
        "waveshare_work TRAIN scenes ONLY (14/17). synth_room_1..5 has no "
        "bounding-box ground truth at all (data/DATASET_NOTES.md), so it cannot "
        "train a bbox-regression detector."
    ),
    "MobileNetSSDDetector": (
        "waveshare_work TRAIN scenes ONLY (14/17), same reason as HOGSVMDetector."
    ),
    "MVSTGCNDetector": (
        "waveshare_work TRAIN scenes' ContactFrameDataset (real per-frame contact "
        "labels) + the FULL synth_room_1..5 corpus, chunked "
        "(iter_contact_training_chunks) since a single synthetic session can be "
        "~1.45M timesteps -- too large to materialize. Training builds sliding "
        "windows from ground-truth ContactEvent.actors directly, NOT from a live "
        "detection+homography pipeline (that's only needed at inference time)."
    ),
    "ThermoX3DDetector": (
        "Same corpus composition as MVSTGCNDetector: waveshare_work TRAIN scenes "
        "+ full synth_room_1..5, chunked."
    ),
}

KNOWN_LIMITATIONS = {
    "MVSTGCNDetector": (
        "Project owner's assessment: the real-deployment homography calibration "
        "approach fuses two cameras' views onto a third camera's image plane "
        "rather than rectifying each camera independently to the true floor "
        "plane, so inter-actor distances derived from it are not physically "
        "meaningful in the way this model's epsilon_m/delta_m thresholds assume. "
        "Evaluation here additionally used a synthetic (not on-site-calibrated) "
        "homography, since none exists yet for waveshare_work. Weak "
        "contact-detection performance is an expected consequence of the "
        "positional input signal, not evidence against the trained weights "
        "themselves -- do not use this checkpoint's contact accuracy as a "
        "signal for retraining decisions without first fixing floor-plane "
        "calibration."
    ),
}

INPUT_OUTPUT_NOTES = {
    "FireSVMDetector": "Input: 6-D feature vector [max_temp, mean_temp, std_temp, area, skewness, kurtosis] from Otsu-segmented hottest blob. Output: FireAlert (level, confidence).",
    "HOGSVMDetector": "Input: HOG descriptor of a sliding-window patch (window_size, multi-scale pyramid). Output: list[Detection] after per-scale NMS.",
    "MobileNetSSDDetector": "Input: (1,1,H,W) float32 thermal frame in degC, normalized internally. Output: list[Detection] after anchor decode + NMS.",
    "MVSTGCNDetector": "Input: ThreeViewFrames (3 raw Frames) + optional pre-computed detections. Internally builds a (1,T=16,N=max_actors,7) node-feature / (1,T,N,2) position / (1,T,N) mask window. Output: ContactEvent.",
    "ThermoX3DDetector": "Input: ThreeViewFrames, buffered into a (1,3,T=16,H,W) volume, globally normalized. Output: ContactEvent.",
}


def _load_checkpoint_payload(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"  WARNING: failed to read {path}: {e}")
        return None


def collect_eval_rows(detector_name: str) -> list[dict]:
    rows = []
    for pat in ("full_corpus_eval_*.json", "eval_*smoketest*.json"):
        for path in sorted(REPORTS_DIR.glob(pat)):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for r in payload.get("results", []):
                if r.get("detector_name") == detector_name:
                    rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", default=str(_REPO_ROOT / "checkpoints_full_corpus"))
    ap.add_argument("--onnx-manifest", default=str(REPORTS_DIR / "onnx_export_manifest.json"))
    args = ap.parse_args()

    checkpoints_root = Path(args.checkpoints)
    manifest = {}
    manifest_path = Path(args.onnx_manifest)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    n_written = 0
    if not checkpoints_root.is_dir():
        print(f"No checkpoints dir at {checkpoints_root}")
        return 1

    for algo_dir in sorted(checkpoints_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        for ckpt_path in sorted(algo_dir.glob("*.thalg")):
            payload = _load_checkpoint_payload(ckpt_path)
            if payload is None:
                continue
            class_name = payload.get("class_name", "")
            short_name = class_name.rsplit(".", 1)[-1]

            eval_rows = collect_eval_rows(short_name)
            onnx_info = manifest.get(short_name)

            sidecar = {
                "class_name": class_name,
                "checkpoint_path": str(ckpt_path.resolve()),
                "format_version": payload.get("format_version"),
                "library_version": payload.get("library_version"),
                "sensor_profile_name": payload.get("sensor_profile_name"),
                "hyperparameters": payload.get("params", {}),
                "training_data": TRAINING_DATA_NOTES.get(short_name, "See data/DATASET_NOTES.md"),
                "input_output": INPUT_OUTPUT_NOTES.get(short_name, ""),
                "known_limitations": KNOWN_LIMITATIONS.get(short_name),
                "onnx_export": onnx_info,
                "evaluated_variants": eval_rows,
                "how_to_load": (
                    f"from thermal_algorithms.core.checkpoints import CheckpointRegistry\n"
                    f"from {class_name.rsplit('.', 1)[0]} import {short_name}\n"
                    f"registry = CheckpointRegistry(root={str(checkpoints_root)!r})\n"
                    f"det = registry.load({short_name}, profile_name={payload.get('sensor_profile_name')!r})"
                ),
            }

            sidecar_path = ckpt_path.with_suffix(".json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
            print(f"  {short_name} ({ckpt_path.stem}) -> {sidecar_path}")
            n_written += 1

            if onnx_info:
                onnx_sidecar_path = Path(onnx_info["onnx_path"]).with_suffix(".json")
                onnx_sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
                print(f"    also -> {onnx_sidecar_path}")

    print(f"\nWrote {n_written} checkpoint metadata sidecar(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
