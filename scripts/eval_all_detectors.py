#!/usr/bin/env python3
"""Evaluate every full-corpus-trained detector on the exact same held-out
waveshare test split, with accuracy/precision/recall/F1/IoU (where
applicable) and mean per-frame inference latency (GPU for the torch
detectors, CPU for the sklearn ones -- there is no meaningful way to force
scikit-learn's SVC decision function onto a GPU).

The held-out split is rebuilt via the exact same function/seed
train_full_corpus.py used (session_train_test_split, default seed=0) so the
test scenes here are identical to what training excluded -- this is what
makes "the same test set for every category" true: fire, human, and contact
are all evaluated against the SAME 3 waveshare scenes.

Writes a JSON file consumed by scripts/generate_full_report.py.

Usage::

    python scripts/eval_all_detectors.py
    python scripts/eval_all_detectors.py --checkpoints checkpoints_baseline_waveshare --out reports/eval_baseline_smoketest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector  # noqa: E402
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.training import (  # noqa: E402
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    PERSON_CLASS_ID,
    evaluate_contact_timed,
    evaluate_fire_timed,
    evaluate_human_timed,
    session_train_test_split,
)

try:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

DATA_ROOT = _REPO_ROOT.parent / "data"


def build_waveshare_test_split(waveshare_index: DatasetIndex) -> tuple[set[str], set[str]]:
    """Mirrors scripts/train_full_corpus.py:build_waveshare_split exactly
    (same function, same default seed=0) -- must never diverge from it, or
    "same test set" stops being true."""
    train_sessions, test_sessions = session_train_test_split(waveshare_index.labeled_sessions())
    return {s.scene for s in train_sessions}, {s.scene for s in test_sessions}


def attach_mvstgcn_inference_deps(det: MVSTGCNDetector, registry: CheckpointRegistry) -> None:
    """MVSTGCNDetector.predict() needs a live human_detector (to produce
    per-camera detections) and a homography (to project them to the floor
    plane) -- neither is part of its own checkpoint (train_full_corpus.py
    constructs it with homography=None; actor positions during *training*
    come straight from ground-truth ContactEvent.actors, not from a live
    detection+projection pipeline, so training never needed either).

    No real on-site Hot-Point Calibration homography exists yet for any
    waveshare_work scene (see data/DATASET_NOTES.md) -- same synthetic
    homography convention used throughout examples/*.py. This means
    MVSTGCN's evaluated contact decisions here reflect the GCN body's
    classification given SYNTHETIC-homography-derived actor positions, not
    true field-calibrated localization; report this caveat alongside the
    numbers, don't present it as a real-deployment result.
    """
    from examples.utils import make_synthetic_homographies
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector

    det._homography = make_synthetic_homographies(det.sensor_profile)
    try:
        det._human_detector = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")
    except Exception:
        from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
        det._human_detector = registry.load(HOGSVMDetector, profile_name="Waveshare_26984")


class _PreprocessedContactDetector:
    """evaluate_contact_timed has no preprocessor hook of its own (matching
    Trainer.evaluate_contact_detection) -- wrap the detector instead."""

    def __init__(self, inner, pre, name: str) -> None:
        self._inner = inner
        self._pre = pre
        self.name = name

    def predict(self, frames):
        return self._inner.predict(tuple(self._pre.predict(f) for f in frames))

    def reset(self) -> None:
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--checkpoints", default=str(_REPO_ROOT / "checkpoints_full_corpus"))
    ap.add_argument("--out", default=str(_REPO_ROOT / "reports" / "full_corpus_eval_baseline.json"))
    ap.add_argument("--variant", default="fp32_baseline")
    ap.add_argument(
        "--only", nargs="*", default=None,
        help="Restrict to these detector class names (e.g. --only HOGSVMDetector). "
             "HOGSVMDetector's brute-force multi-scale sliding window is ~2.8s/frame "
             "on CPU -- run it as its own job in parallel with the fast detectors "
             "rather than serializing everything behind it.",
    )
    args = ap.parse_args()
    only = set(args.only) if args.only else None

    data_root = Path(args.data_root)
    waveshare_index = DatasetIndex(data_root / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    train_scenes, test_scenes = build_waveshare_test_split(waveshare_index)
    print(f"Test scenes ({len(test_scenes)}): {sorted(test_scenes)}")

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit()

    registry = CheckpointRegistry(root=args.checkpoints)
    results = []

    def try_load(cls, profile_name):
        try:
            return registry.load(cls, profile_name=profile_name)
        except Exception as e:
            print(f"  SKIP {cls.__name__}: {e}")
            return None

    def wanted(name: str) -> bool:
        return only is None or name in only

    # --- Fire (invariant checkpoint) ---
    if wanted("FireSVMDetector"):
        fire_ds = FireFrameDataset(waveshare_index, scenes=test_scenes)
        det = try_load(FireSVMDetector, None)
        if det is not None:
            r = evaluate_fire_timed(det, fire_ds, preprocessor=preprocessor, variant=args.variant)
            results.append(r)
            print(r.to_dict())

    # --- Human ---
    human_ds = FrameLevelDataset(
        waveshare_index, scenes=test_scenes, class_filter=[PERSON_CLASS_ID], include_negative_frames=True,
    )
    if wanted("HOGSVMDetector"):
        det = try_load(HOGSVMDetector, "Waveshare_26984")
        if det is not None:
            r = evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant=args.variant)
            results.append(r)
            print(r.to_dict())

    if wanted("MobileNetSSDDetector"):
        if _TORCH_OK:
            det = try_load(MobileNetSSDDetector, "Waveshare_26984")
            if det is not None:
                r = evaluate_human_timed(det, human_ds, preprocessor=preprocessor, variant=args.variant)
                results.append(r)
                print(r.to_dict())
        else:
            print("  SKIP MobileNetSSDDetector: torch not available")

    # --- Contact ---
    contact_ds = ContactFrameDataset(waveshare_index, scenes=test_scenes)
    for label, cls in (("MVSTGCNDetector", MVSTGCNDetector), ("ThermoX3DDetector", ThermoX3DDetector)):
        if not wanted(label):
            continue
        if not _TORCH_OK:
            print(f"  SKIP {label}: torch not available")
            continue
        det = try_load(cls, "Waveshare_26984")
        if det is None:
            continue
        if label == "MVSTGCNDetector":
            attach_mvstgcn_inference_deps(det, registry)
        wrapped = _PreprocessedContactDetector(det, preprocessor, label)
        r = evaluate_contact_timed(wrapped, contact_ds, variant=args.variant, detector_name=label)
        results.append(r)
        print(r.to_dict())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoints_dir": str(Path(args.checkpoints).resolve()),
        "test_scenes": sorted(test_scenes),
        "train_scenes": sorted(train_scenes),
        "results": [r.to_dict() for r in results],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
