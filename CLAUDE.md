# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

**Smart Thermal System for Patient Safety Monitoring** — a privacy-preserving edge system for psychiatric wards (Beer Sheva Mental Health Center). Three MLX90640 32×24 thermal sensors on a Raspberry Pi 5, detecting four anomaly scenarios: fire ignition, physical contact between patients, violent/sexual behaviour, and restricted-area breach. No optical images — thermal only, preserving patient privacy.

Students: Guy Chen, Yaniv Blau, Roy Lieberman. Supervisor: Or Zilberberg. Advisor: Dr. Oshrit Hoffer (Afeka College of Engineering, Tel-Aviv).

**Current phase:** Active evaluation on the **Waveshare** dataset. A labeled dataset exists at `datasets/waveshare_work`; the `scripts/` directory holds the benchmark campaign — human / fire / contact detectors are trained and tuned, with results written to `reports/` (JSON + LaTeX/PDF). The contact pipeline is the most active area (versioned `eval_waveshare_contact_v3..v10`). The §5.3 report tables are now Waveshare-based, so **`WAVESHARE_26984` (80×62) is the primary working profile**; `MLX90640` (32×24) is retained as the original target hardware.

---

## Commands

```bash
# Install (editable, with dev tools: pytest, pytest-cov, ruff, mypy, openpyxl)
pip install -e ".[dev]"

# Run the full test suite (skimage / torch tests auto-skip if those deps are missing)
python -m pytest                         # 460 tests across 21 files

# Run a single test file / single test
python -m pytest tests/test_pipeline.py -v
python -m pytest tests/test_pipeline.py::TestRestrictedAreaPath::test_human_detected_triggers_alert -v

# Lint + typecheck (config in pyproject.toml: line-length 100, target py310)
ruff check .
mypy thermal_algorithms

# Demo scripts — all fall back to synthetic data automatically when no dataset is present
python examples/demo_pipeline.py
python examples/demo_fire_detection.py
python examples/demo_contact_geometric.py

# Re-run a benchmark (reads datasets/waveshare_work, writes reports/<name>_results.json)
python scripts/eval_waveshare_human.py
python scripts/eval_waveshare_fire.py
python scripts/eval_waveshare_contact_v10.py
```

**Tests requiring optional deps (auto-skipped via `pytest.importorskip`):**
- `tests/test_adaptive_threshold.py`, `tests/test_hog_svm.py` — require `scikit-image`
- `tests/test_mobilenet_ssd.py`, `tests/test_contact_mv_stgcn.py`, `tests/test_contact_thermo_x3d.py` — require `torch`

---

## Architecture

### Package layout

```
thermal_algorithms/
├── core/               base ABC, frozen dataclasses, sensor profiles, checkpoints, CSV I/O
├── preprocessing/      TatenoPipeline (Gaussian smooth → background subtract → L1 residual)
├── human_detection/    3 alternatives: AdaptiveThreshold / HOG-SVM / MobileNet-SSD
├── fire_detection/     2 alternatives: OtsuFireDetector / FireSVMDetector
├── contact_detection/  3 alternatives: Geometric / MV-STGCN / Thermo-X3D
│   └── multi_view/     shared: homography.py, tracker.py, fusion.py
├── training/           datasets, label I/O, metrics, trainer
└── pipeline.py         ThermalPipeline — runtime integration (Figure 5 flow)

examples/               demo scripts (all fall back to synthetic data)
  utils.py              shared helpers: load_frames, find_session, make_synthetic_homographies
scripts/                applied-research workflow: dataset prep, benchmarks, plots (see below)
image_annotator/        Tkinter YOLO annotation tool for the Waveshare dataset (own CLAUDE.md)
reports/                benchmark outputs — *_results.json + *_report.{tex,pdf}
tests/                  one file per module; 460 tests across 21 files
```

### The `scripts/` workflow (where most current work happens)

`scripts/` is the applied-research layer that consumes the `thermal_algorithms` package against the real dataset. Conventions:

- **Dataset root** is `datasets/waveshare_work` (git-ignored, "layout B": one dir per *scene*, with `chN_frames/`, `chN_raw_data.npz`, YOLO `.txt`, and `contact_labels.csv`). Scripts add the repo root to `sys.path` and load via `DatasetIndex(root, sensor_profile=WAVESHARE_26984)`.
- **`eval_waveshare_*`** scripts run a detector end-to-end on the dataset, print a readable table, and dump `reports/<name>_results.json`. The `*.tex`/`*.pdf` reports in `reports/` are the figures/tables for §5.3.
- **Contact eval is versioned** (`eval_waveshare_contact`, then `_v3_variants` … `_v10`). Later versions **import earlier ones as modules** (e.g. `v10` imports `geo`, `vv`, `v8`) to reuse split logic and helpers — don't rename or break the public functions/constants in an earlier version without checking who imports it.
- **`reorganize_*`, `consolidate_*`, `fix_*`, `reconcile_*`, `truncate_*`** are one-shot dataset-maintenance scripts. **`plot_*`** and **`visualize_*`** generate report figures.
- **`empty_room` / `calibrate_room`** scenes are special: empty frames calibrate the per-channel Tateno background (and supply true negatives); `calibrate_room` drives homography self-calibration. Keep calibration and evaluation frames disjoint — scripts split sequentially (no temporal leakage).

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
# Typical evaluation flow matching §5.3 report tables.
# The real, worked-out versions of this live in scripts/eval_waveshare_*.py.
index  = DatasetIndex("datasets/waveshare_work", sensor_profile=WAVESHARE_26984)

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

## Detector training / calibration inputs

(The Waveshare dataset now satisfies most of these; this is the reference for what each detector's `fit()`/calibration consumes.)

- **FireSVMDetector.fit()** — needs `FireFrameDataset` with annotated fire frames
- **HOGSVMDetector.fit()** — needs `FrameLevelDataset` with person bbox labels
- **MobileNetSSDDetector.fit()** — same, with enough data for DL training
- **MVSTGCNDetector.fit()** / **ThermoX3DDetector.fit()** — need `ContactFrameDataset` with `contact_labels.csv`
- **Homography calibration** — `detector.calibrate_homography(marker_correspondences)` requires ≥4 heated targets at known floor coordinates per camera
- **OtsuFireDetector** and **GeometricContactDetector** are rule-based — no training needed, but thresholds (`t_ign`, `t_fire`, `delta_m`) will need tuning against real data

---

## Key open items

1. **Threshold calibration** — `OtsuFireDetector` defaults: `t_ign=45°C`, `t_fire=60°C`, `a_limit≈3%` of frame. These were derived from initial EDA and will need validation against the full dataset.
2. **Homography for contact detection** — demos still use synthetic homographies. For real data there are now two routes: `multi_view/homography.calibrate_floor_homographies_from_tracks` (RANSAC+DLT self-calibration from the shared `calibrate_room` track — ε/δ end up in cam0 floor-pixel units, not metres) and the annotator's manual `🎯 Homography Calibration` dialog (writes `homography_calibration.npz` with `h1/h2/h3`). The on-site metric Hot-Point Calibration is still the gold standard.
3. **Restricted area zones** — the pipeline supports `restricted=True/False` globally; per-camera restriction or floor-polygon containment checks are not yet implemented.
