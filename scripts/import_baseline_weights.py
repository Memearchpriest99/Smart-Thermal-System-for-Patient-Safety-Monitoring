#!/usr/bin/env python3
"""Import the surviving `Weights/` files into proper `.thalg` checkpoints.

Context: waveshare_work was trained once, in full, before its bbox/contact
annotations were lost. The five resulting weight files
(`fire_svm.pkl`, `hog_svm.pkl`, `mobilenet_ssd.pt`, `mv_stgcn.pt`,
`thermo_x3d.pt`) survive in a `Weights/` folder at the project root (a
sibling of this repo and `data/`). This script registers them as a
"baseline" checkpoint set via `CheckpointRegistry`, so Phase 3 can fine-tune
from them instead of training from scratch.

Two different file shapes were found on inspection:

* `fire_svm.pkl` / `hog_svm.pkl` are, byte-for-byte, already the exact
  payload `ThermalAlgorithm.save()` produces (`format_version`, `class_name`,
  `params`, `state`, `is_fitted`) — confirmed by unpickling and comparing
  keys, not assumed from the `.pkl` extension. `<Class>.load()` doesn't care
  about file extension, so these load directly.
* `mobilenet_ssd.pt` / `mv_stgcn.pt` / `thermo_x3d.pt` are raw
  `torch.save(...)` dumps of just the `_state_dict()` sub-dict (e.g.
  `{"model_state": ..., "input_size": ...}`), NOT the full outer payload —
  confirmed the same way. These need a live detector instance constructed
  first, `_load_state_dict()` called directly, then `.save()`/`register()`.

Known limitation: `mv_stgcn.pt` has no `"homography"` key (the manifest's
`homography` section is calibration metadata, not the actual H matrices), so
the imported `MVSTGCNDetector` has `homography=None` — its multi-view fusion
front-end will produce zero actors until a real homography is supplied
separately (see `examples/calibrate_homography.py`).

Usage::

    python scripts/import_baseline_weights.py
    python scripts/import_baseline_weights.py --weights-dir ../Weights --out checkpoints_baseline_waveshare
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
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector  # noqa: E402
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector  # noqa: E402
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402

DEFAULT_WEIGHTS_DIR = _REPO_ROOT.parent / "Weights"
DEFAULT_OUT = _REPO_ROOT / "checkpoints_baseline_waveshare"


def import_full_payload_checkpoint(cls, path: Path, registry: CheckpointRegistry):
    """For files that are already a full ThermalAlgorithm.save() payload."""
    det = cls.load(path)
    saved_to = registry.register(det)
    print(f"  {cls.__name__:<22} <- {path.name:<20} -> {saved_to}")
    return det


def import_raw_state_checkpoint(det, path: Path, registry: CheckpointRegistry):
    """For files that are just the _state_dict() sub-dict (raw torch.save)."""
    import torch

    raw_state = torch.load(path, map_location="cpu", weights_only=False)
    det._load_state_dict(raw_state)
    saved_to = registry.register(det)
    print(f"  {type(det).__name__:<22} <- {path.name:<20} -> {saved_to}")
    return det


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    if not weights_dir.is_dir():
        raise SystemExit(f"Weights dir not found: {weights_dir}")

    registry = CheckpointRegistry(root=args.out)
    print(f"Importing {weights_dir} -> {registry.root}\n")

    import_full_payload_checkpoint(FireSVMDetector, weights_dir / "fire_svm.pkl", registry)
    import_full_payload_checkpoint(HOGSVMDetector, weights_dir / "hog_svm.pkl", registry)

    import_raw_state_checkpoint(
        MobileNetSSDDetector(sensor_profile=WAVESHARE_26984),
        weights_dir / "mobilenet_ssd.pt",
        registry,
    )
    import_raw_state_checkpoint(
        MVSTGCNDetector(sensor_profile=WAVESHARE_26984, homography=None),
        weights_dir / "mv_stgcn.pt",
        registry,
    )
    import_raw_state_checkpoint(
        ThermoX3DDetector(sensor_profile=WAVESHARE_26984),
        weights_dir / "thermo_x3d.pt",
        registry,
    )

    print("\nAvailable checkpoints:")
    for algo, profile in registry.list_available():
        print(f"  {algo} / {profile or 'invariant'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
