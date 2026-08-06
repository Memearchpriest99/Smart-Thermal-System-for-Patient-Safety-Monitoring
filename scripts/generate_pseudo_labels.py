#!/usr/bin/env python3
"""Generate pseudo (silver) person-detection labels for waveshare_work.

Only 1 of 17 scenarios (``1_man_run``) currently has real YOLO annotations.
Uses the baseline `MobileNetSSDDetector` imported by
`scripts/import_baseline_weights.py` (trained on the full dataset before
those annotations were lost) to regenerate approximate boxes for the other
16 scenarios, saved as a single JSON file — NOT written into the real
dataset tree, and NOT treated as ground truth anywhere downstream.

Usage::

    python scripts/generate_pseudo_labels.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector  # noqa: E402
from thermal_algorithms.training.datasets import DatasetIndex  # noqa: E402
from thermal_algorithms.training.pseudo_labels import (  # noqa: E402
    generate_pseudo_person_labels,
    save_pseudo_labels,
)

DEFAULT_DATA_ROOT = _REPO_ROOT.parent / "data" / "waveshare_work"
DEFAULT_CHECKPOINTS = _REPO_ROOT / "checkpoints_baseline_waveshare"
DEFAULT_OUT = _REPO_ROOT / "data_artifacts" / "waveshare_pseudo_person_labels.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--checkpoints", default=str(DEFAULT_CHECKPOINTS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    print(f"Loading baseline MobileNetSSDDetector from {args.checkpoints} ...")
    registry = CheckpointRegistry(root=args.checkpoints)
    detector = registry.load(MobileNetSSDDetector, profile_name="Waveshare_26984")

    print(f"Scanning {data_root} ...")
    index = DatasetIndex(data_root, sensor_profile=WAVESHARE_26984, fps=8.0)
    print(f"  {len(index.sessions)} sessions, "
          f"{sum(1 for s in index.sessions if s.has_labels)} with real annotations")

    print("Running inference (skipping frames that already have real annotations) ...")
    labels = generate_pseudo_person_labels(index, detector, skip_annotated=True)

    n_frames = sum(len(ch_boxes) for scene in labels.values() for ch_boxes in scene.values())
    n_positive = sum(
        1
        for scene in labels.values()
        for ch_boxes in scene.values()
        for boxes in ch_boxes.values()
        if boxes
    )
    print(f"\nGenerated pseudo-labels for {n_frames} (scene, channel, frame) entries "
          f"across {len(labels)} scenes; {n_positive} have >=1 detected box "
          f"({100 * n_positive / n_frames:.1f}%).")

    out_path = Path(args.out)
    save_pseudo_labels(labels, out_path)
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
