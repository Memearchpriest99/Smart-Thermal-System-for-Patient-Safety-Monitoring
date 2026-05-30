# examples

Runnable scripts demonstrating each algorithm. All scripts fall back to **synthetic data** automatically when the real dataset is not available, so they work out of the box.

## Running

```bash
# From the project root:
python examples/demo_tateno_pipeline.py      # preprocessing pipeline
python examples/demo_adaptive_threshold.py   # classical human detection
python examples/demo_hog_svm.py              # HOG + SVM human detection
python examples/demo_fire_detection.py       # fire detection (Otsu pipeline)
python examples/demo_contact_geometric.py    # geometric contact detection
python examples/demo_pipeline.py             # full end-to-end ThermalPipeline
python examples/calibrate_homography.py      # guided homography calibration
```

All outputs (PNG figures, calibration files) are written to `outputs/`.

## Configuring real data

Each script has a `DATASET_ROOT` / `RECORDING_DIR` constant at the top. Set it to your dataset path to use real sensor data instead of the synthetic fallback.

## calibrate_homography.py

Guided interactive script for the Hot-Point Calibration procedure required by `GeometricContactDetector` (and any multi-view algorithm that needs `HomographyMatrices`).

**Procedure:**
1. Mount the 3 cameras at their final room positions
2. Place ≥ 4 heated markers (hot water containers, hand warmers) at known floor coordinates
3. Set `MARKER_WORLD_POSITIONS` and `RECORDING_DIR` in the script
4. Run the script — click each marker in each camera view when prompted
5. Calibration is saved to `outputs/homography_calibration.npz` and `.thalg`

**Loading the result:**
```python
import numpy as np
from thermal_algorithms.core.types import HomographyMatrices
cal = np.load("outputs/homography_calibration.npz")
H = HomographyMatrices(h1=cal["h1"], h2=cal["h2"], h3=cal["h3"])
```

## utils.py

Shared helpers used by all demo scripts. Import from here to avoid duplication:

```python
from examples.utils import load_frames, find_session, make_synthetic_homographies, make_synthetic_triplet
```
