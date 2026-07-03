# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

**Smart Thermal System for Patient Safety Monitoring** — a privacy-preserving edge system for psychiatric wards (Beer Sheva Mental Health Center). Three MLX90640 32×24 thermal sensors on a Raspberry Pi 5, detecting four anomaly scenarios: fire ignition, physical contact between patients, violent/sexual behaviour, and restricted-area breach. No optical images — thermal only, preserving patient privacy.

Students: Guy Chen, Yaniv Blau, Roy Lieberman. Supervisor: Or Zilberberg. Advisor: Dr. Oshrit Hoffer (Afeka College of Engineering, Tel-Aviv).

**Current phase:** Dataset annotation (YOLO format). Algorithms and training infrastructure are complete; awaiting labeled data to train ML models and run §5.3 benchmarks.

---

## Commands

```bash
# Run the full working test suite (skimage / torch tests are gated separately)
python -m pytest tests/test_types.py tests/test_sensor_profile.py tests/test_base.py \
  tests/test_checkpoints.py tests/test_tateno_pipeline.py tests/test_otsu_pipeline.py \
  tests/test_fire_svm.py tests/test_contact_multiview.py tests/test_contact_geometric.py \
  tests/test_contact_mv_stgcn.py tests/test_contact_thermo_x3d.py tests/test_metrics.py \
  tests/test_label_io_extended.py tests/test_fire_contact_datasets.py \
  tests/test_trainer.py tests/test_pipeline.py tests/test_acquisition.py

# Run a single test file
python -m pytest tests/test_pipeline.py -v

# Run a single test by name
python -m pytest tests/test_pipeline.py::TestRestrictedAreaPath::test_human_detected_triggers_alert -v

# Run demo scripts (require dataset at /sessions/magical-youthful-euler/mnt/dataset;
# all fall back to synthetic data automatically when the path is missing)
python examples/demo_fire_detection.py
python examples/demo_contact_geometric.py
python examples/demo_pipeline.py
```

**Tests that require optional deps (skip if not installed):**
- `tests/test_adaptive_threshold.py`, `tests/test_hog_svm.py` — require `scikit-image`
- `tests/test_mobilenet_ssd.py`, `tests/test_contact_mv_stgcn.py`, `tests/test_contact_thermo_x3d.py` — require `torch` (auto-skipped via `pytest.importorskip`)

---

## Architecture

### Package layout

```
thermal_algorithms/
├── core/               base ABC, frozen dataclasses, sensor profiles, checkpoints, CSV I/O
├── acquisition/        live capture from the Waveshare MI48 module — wiki SPI/I2C pipeline
│                       via pysenxor; hardware imports are lazy (testable off-Pi)
├── preprocessing/      TatenoPipeline (Gaussian smooth → background subtract → L1 residual)
├── human_detection/    3 alternatives: AdaptiveThreshold / HOG-SVM / MobileNet-SSD
├── fire_detection/     2 alternatives: OtsuFireDetector / FireSVMDetector
├── contact_detection/  3 alternatives: Geometric / MV-STGCN / Thermo-X3D
│   └── multi_view/     shared: homography.py, tracker.py, fusion.py
├── training/           datasets, label I/O, metrics, trainer
└── pipeline.py         ThermalPipeline — runtime integration (Figure 5 flow)

examples/               demo scripts (all fall back to synthetic data)
  utils.py              shared helpers: load_frames, find_session, make_synthetic_homographies
tests/                  one file per module; 363 tests passing
```

### The universal base contract

Every algorithm inherits `ThermalAlgorithm` (`core/base.py`):
- `fit(X, y=None)` — training or calibration (returns `self`)
- `predict(X)` — inference
- `save(path)` / `load(path)` — pickle-backed checkpoints
- `reset()` — clears temporal state (stateful detectors override this)
- Class-level: `name` (registry key), `is_trainable`, `resolution_behavior` (`invariant` | `parameterized` | `fixed`)

### Data types (core/types.py)

All algorithms speak these frozen dataclasses:
- `Frame(data: np.ndarray[H,W], timestamp, camera_id)` — one thermal frame
- `Detection(bbox: (x,y,w,h), score, class_id, camera_id)` — one bounding box
- `FireAlert(level: FireLevel, timestamp, blob_features, confidence)`
- `ContactEvent(actors, pairs_in_contact, timestamp, confidence)`
- `HomographyMatrices(h1, h2, h3)` — 3×3 per camera, image→floor plane

### YOLO class convention (user-confirmed)

```
class 0 = fire / ignition source
class 1 = person
```
Defined as `FIRE_CLASS_ID = 0` and `PERSON_CLASS_ID = 1` in `training/label_io.py`.

Contact labels are **not** bounding boxes. They live in a per-session `contact_labels.csv`:
```
frame_idx,contact
0,0
5,1
6,1
```
Only annotated frames appear; missing frames are excluded from training.

### The training stack

```python
# Typical evaluation flow matching §5.3 report tables
index  = DatasetIndex("data/", sensor_profile=MLX90640)

# Human detection — FrameLevelDataset with class_filter=[PERSON_CLASS_ID]
# Fire detection  — FireFrameDataset (wraps YOLO class 0, produces FireAlert)
# Contact         — ContactFrameDataset (reads contact_labels.csv, produces ContactEvent)

results = Trainer.evaluate_fire_detection(detector, fire_ds, mode="proc")
print(format_scenario_table(results, include_iou=True))
```

`Trainer` methods: `evaluate_human_detection`, `evaluate_fire_detection`, `evaluate_contact_detection`, `evaluate_preprocessing`, `fit_and_evaluate`. All call `detector.reset()` at session boundaries automatically.

`Trainer.evaluate_preprocessing` returns `(mean_raw_sbr, mean_processed_sbr)` — the report achieved 5.61× (1.21 → 6.80).

### The runtime pipeline

`ThermalPipeline` matches Figure 5's flow diagram exactly:

```python
pipeline = ThermalPipeline(
    preprocessor=...,
    human_detector=...,
    fire_detector=...,
    contact_detector=...,
    restricted=False,           # True = restricted-area mode (human detection only)
    fire_cooldown_s=30.0,
    contact_cooldown_s=30.0,
    restricted_area_cooldown_s=30.0,
)
result = pipeline.process(frame0, frame1, frame2)
# result.alerts: tuple[Alert, ...] — empty = all-clear
# result.fire_alarm, result.contact_alarm, result.restricted_area_alarm: bool
```

**Restricted mode** (`restricted=True`): only human detection runs; fire and contact detectors are completely skipped. Any human detected → `AlertType.RESTRICTED_AREA`. This matches the YES branch of the "Restricted?" gate in Figure 5.

**Full mode** (`restricted=False`): fire detection runs per-camera (alarm if any camera alarms), contact detection runs on the 3-view bundle. `AlertType.FIRE` or `AlertType.CONTACT` are produced.

Stateful detectors are reset across calls automatically — call `pipeline.reset()` between independent sequences.

### Sensor profiles

Two profiles in `core/sensor_profile.py`:
- `MLX90640`: 32×24 px, 55°×35° FOV, 8 Hz, noise floor 1.5°C
- `WAVESHARE_26984`: 80×62 px, 60°×45° FOV, 8 Hz, noise floor 0.7°C

`resolution_behavior = "parameterized"` algorithms auto-scale kernel sizes via `profile.physical_pixel_size_m(distance_m)`. `"fixed"` algorithms (MobileNet-SSD, Thermo-X3D) need one checkpoint per profile.

### Contact detection multi-view stack

All three contact detectors share:
- `multi_view/homography.py` — DLT+SVD solver (`solve_homography_from_markers`)
- `multi_view/tracker.py` — Kalman CV tracker (`PerCameraTracker`) for I2C phase-shift correction (Camera 3 reads ~50 ms after Camera 1)
- `multi_view/fusion.py` — foot-point projection + cross-camera validation (N=1 discard, N=3 outlier removal) + clustering → `ActorPosition`

Key parameter: `epsilon_m` (clustering threshold, ~0.15–0.20 m reflecting homography accuracy) is distinct from `delta_m` (contact threshold, ~0.5–0.6 m). These must be tuned independently.

### Checkpoints

`CheckpointRegistry` (`core/checkpoints.py`) maps `(algorithm_name, profile_name)` to `.thalg` files (pickle). Layout:
```
checkpoints/<algorithm_name>/<profile_name>.thalg
checkpoints/<algorithm_name>/_default.thalg   # invariant algorithms
```

---

## What requires data before it works

- **FireSVMDetector.fit()** — needs `FireFrameDataset` with annotated fire frames
- **HOGSVMDetector.fit()** — needs `FrameLevelDataset` with person bbox labels
- **MobileNetSSDDetector.fit()** — same, with enough data for DL training
- **MVSTGCNDetector.fit()** / **ThermoX3DDetector.fit()** — need `ContactFrameDataset` with `contact_labels.csv`
- **Homography calibration** — `detector.calibrate_homography(marker_correspondences)` requires ≥4 heated targets at known floor coordinates per camera
- **OtsuFireDetector** and **GeometricContactDetector** are rule-based — no training needed, but thresholds (`t_ign`, `t_fire`, `delta_m`) will need tuning against real data

---

## Key open items

1. **Threshold calibration** — `OtsuFireDetector` defaults: `t_ign=45°C`, `t_fire=60°C`, `a_limit≈3%` of frame. These were derived from initial EDA and will need validation against the full dataset.
2. **Homography for contact detection** — synthetic homographies are used in all demos; real H matrices require the on-site Hot-Point Calibration procedure.
3. **Restricted area zones** — the pipeline supports `restricted=True/False` globally; per-camera restriction or floor-polygon containment checks are not yet implemented.
