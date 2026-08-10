# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

**Smart Thermal System for Patient Safety Monitoring** — a privacy-preserving edge system for psychiatric wards (Beer Sheva Mental Health Center). Three MLX90640 32×24 thermal sensors on a Raspberry Pi 5, detecting four anomaly scenarios: fire ignition, physical contact between patients, violent/sexual behaviour, and restricted-area breach. No optical images — thermal only, preserving patient privacy.

Students: Guy Chen, Yaniv Blau, Roy Lieberman. Supervisor: Or Zilberberg. Advisor: Dr. Oshrit Hoffer (Afeka College of Engineering, Tel-Aviv).

**Current phase:** full-corpus + balanced-regime training and evaluation (see "Key open items" below for live status per task). Dataset annotation is done — `waveshare_work` has complete real YOLO/contact labels across all 17 scenarios, and the multi-source training layer for `synth_room_1..5` is built and tested. `GlobalNormPreprocessor` won the preprocessing comparison against `TatenoPipeline` (`reports/preprocessing_comparison_results.json`) and is what `scripts/train_full_corpus.py` uses. See `../data/DATASET_NOTES.md` for the full, verified-against-disk dataset writeup (schemas, class ratios, why `room-1` is excluded).

**Training run status (2026-08-08):** `scripts/train_full_corpus.py` runs via the Windows scheduled
task `ThermalFullCorpusTraining` (see `run_full_corpus_detached.bat`), not a foreground/Bash
process — background shell tasks in this environment get killed after a bounded duration
regardless of process health, and the run takes many hours. A run was killed intentionally before
reaching the contact stage: `MVSTGCNDetector`/`ThermoX3DDetector` each default to `n_epochs=30`
*inside a single `fit()` call*, and the chunked full-corpus contact-training loop calls `fit()`
once per ~5000-frame chunk across ~6200+ chunks — meaning the old default would have run ~30
epochs redundantly on every chunk (measured: 36.1s/chunk for MVSTGCN alone → ~62.2h extrapolated,
before ThermoX3D, against the scheduled task's 72h execution limit). Fixed via a new
`--contact-epochs-per-chunk` CLI arg (default 1, threaded into both detectors' constructors) — the
chunk count itself now provides corpus coverage instead of multiplying epochs by chunks. Not yet
re-verified with a full end-to-end run; a reboot was requested before the next attempt. No
`checkpoints_full_corpus/` with real full-corpus weights exists yet.

---

## Commands

```bash
# Run the whole suite (testpaths=tests is set in pyproject.toml)
python -m pytest

# Run a single test file
python -m pytest tests/test_pipeline.py -v

# Run a single test by name
python -m pytest tests/test_pipeline.py::TestRestrictedAreaPath::test_human_detected_triggers_alert -v

# Run demo scripts (require dataset at /sessions/magical-youthful-euler/mnt/dataset;
# all fall back to synthetic data automatically when the path is missing)
python examples/demo_fire_detection.py
python examples/demo_contact_geometric.py
python examples/demo_pipeline.py

# Task 3: retrain every trainable detector on the full corpus (synth_room_1..5 + waveshare_work)
python scripts/train_full_corpus.py
python scripts/train_full_corpus.py --skip-contact          # fire+human only, much faster
```

**Tests that require optional deps (skip if not installed):**
- `tests/test_adaptive_threshold.py`, `tests/test_hog_svm.py` — require `scikit-image`
- `tests/test_mobilenet_ssd.py`, `tests/test_contact_mv_stgcn.py`, `tests/test_contact_thermo_x3d.py` — require `torch` (auto-skipped via `pytest.importorskip`)

**Known-failing tests (pre-existing, unrelated to the training-pipeline work):** `tests/test_mobilenet_ssd.py::TestArchitecture::test_waveshare_forward_shapes` and `::TestDetector::test_save_load_roundtrip` fail against the currently-installed `torch==2.13`/`opencv==5.0` — a backbone stride-rounding assumption is off by one pixel, and predict() isn't deterministic across save/load. Everything else passes (603 passed, 2 skipped as of the last full run).

---

## Architecture

### Package layout

```
thermal_algorithms/
├── core/               base ABC, frozen dataclasses, sensor profiles, checkpoints, CSV I/O
├── acquisition/        live capture from the Waveshare MI48 module — wiki SPI/I2C pipeline
│                       via pysenxor; hardware imports are lazy (testable off-Pi)
├── preprocessing/      TatenoPipeline (learned per-pixel background) and GlobalNormPreprocessor
│                       (per-frame scalar background, no calibration) — see "Preprocessing" below
├── human_detection/    3 alternatives: AdaptiveThreshold / HOG-SVM / MobileNet-SSD
├── fire_detection/     2 alternatives: OtsuFireDetector / FireSVMDetector
├── contact_detection/  3 alternatives: Geometric / MV-STGCN / Thermo-X3D
│   └── multi_view/     shared: homography.py, tracker.py, fusion.py
├── training/           datasets (waveshare/YOLO), label I/O, metrics, trainer,
│                       + the synth/room-1 multi-source layer — see below
└── pipeline.py         ThermalPipeline — runtime integration (Figure 5 flow)

examples/               demo scripts (all fall back to synthetic data)
  utils.py              shared helpers: load_frames, find_session, make_synthetic_homographies
scripts/                one-off eval/training drivers; train_full_corpus.py is the Task 3 entrypoint
tests/                  one file per module; 600+ tests passing
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

### Preprocessing: TatenoPipeline vs GlobalNormPreprocessor

`GlobalNormPreprocessor` (`preprocessing/global_norm_pipeline.py`) is a redesign of the annotation
tool's client-side display filter into a real `Preprocessor`: Gaussian smooth → subtract the
**current frame's own scalar mean** (not a learned per-pixel background) → L1 residual. `fit()` is
a genuine no-op — there's no calibration state, unlike `TatenoPipeline`'s per-pixel background
learned from empty-room frames. This was built specifically to A/B against `TatenoPipeline`
(Task 2); `reports/preprocessing_comparison_results.json` (via
`scripts/eval_preprocessing_comparison.py`) is the result — `GlobalNormPreprocessor` won on SBR,
fire accuracy/recall, and contact accuracy, with human detection roughly tied. `scripts/
train_full_corpus.py` uses it as the default preprocessor.

### The multi-source training layer (`training/hdf5_source.py`, `label_join.py`, `multi_source.py`, `split.py`, `full_corpus.py`)

`waveshare_work` and `synth_room_1..5`/`room-1` are structurally different sources and are NOT
unified into one dataset class — see `data/DATASET_NOTES.md` (one level up) for the full,
verified-against-disk writeup. In short:

- **`hdf5_source.py`** reads the synthetic/real-hardware chunked `.h5` files
  (`<date>/cam_{0,1,2}/*.h5`) that `DatasetIndex` doesn't understand at all. `HDF5CameraSession`
  loads lazily (LRU chunk cache) — a full synthetic camera-day is ~1.45M frames, too large to
  materialize. Handles two real acquisition bugs found on disk: unusable per-chunk timestamps
  (falls back to filename-derived wall-clock time, chained off the previous chunk's end) and a
  fps that isn't the same across sessions (inferred from chunk-filename spacing, not hardcoded).
- **`label_join.py`** turns a room's `labels.csv` (event **intervals**, not per-frame) into
  per-frame fire/human/contact booleans via substring rules on `Event_Class` (e.g.
  `Contact_2+Humans+Fire`). No bounding boxes exist in this source, ever.
- **`multi_source.py`** (`HDF5Session`, `MultiSourceContactDataset`, `MultiSourceFireDataset`)
  combines `waveshare_work`'s real bbox/contact examples with the synth sources' presence-only
  examples behind one iterable. These are drop-ins for `Trainer.evaluate_contact_detection`/
  `evaluate_fire_detection` (duck-typed on `.by_session()`/`.scene`) but **not** recognized by
  `Trainer.fit_and_evaluate`'s `isinstance` dispatch — assemble training examples by iterating
  directly instead (see each class's docstring).
- **`split.py`** — session-level (not frame-level) train/test split for real sources, avoiding
  leakage between near-duplicate consecutive frames. Synthetic data is used wholesale for
  training (no split) per the Task 3 instruction.
- **`full_corpus.py`** — streaming helpers (`iter_synth_sessions`, `stream_fire_examples`,
  `iter_contact_training_chunks`) over the *entire* corpus (~31M frame-instances) without ever
  materializing a list. Contact detectors (MV-STGCN, Thermo-X3D) build sliding windows by
  materializing `fit(X, y)` internally, so they must be fed bounded, session-contiguous chunks —
  `iter_contact_training_chunks` / `scripts/train_full_corpus.py`'s chunked loop is the pattern to
  reuse for any future full-corpus training.
- **`pseudo_labels.py`** — infrastructure for silver-labeling a source with no real bboxes via a
  fitted detector's own predictions. Built for `waveshare_work` before its real annotations were
  restored; now superseded there (real ground truth exists for all 17 scenarios) but kept as
  tested infrastructure — e.g. `synth_room_*` could be pseudo-labeled the same way if ever needed.

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

## What requires data before it works — now resolved except homography

All of the below now have real data available (`waveshare_work`'s bboxes/contact labels are
complete across all 17 scenarios; `synth_room_1..5` adds frame-level presence data for fire/
human/contact). `scripts/train_full_corpus.py` is the entrypoint that fits all of them.

- **FireSVMDetector.fit()** / **MVSTGCNDetector.fit()** / **ThermoX3DDetector.fit()** — can train
  on the full corpus (waveshare + all synth rooms; contact detectors need chunked fitting, see
  `iter_contact_training_chunks`).
- **HOGSVMDetector.fit()** / **MobileNetSSDDetector.fit()** — waveshare_work only (synth has no
  bounding boxes, ever).
- **Homography calibration** — `detector.calibrate_homography(marker_correspondences)` requires
  ≥4 heated targets at known floor coordinates per camera. Synthetic homographies are still used
  in demos; `image_annotator`'s Homography Calibration dialog produces a real one per session but
  no real H matrices have been wired into the full-corpus training run yet.
- **OtsuFireDetector**'s `t_ign` is now data-calibrated (`scripts/calibrate_otsu_thresholds.py`,
  2026-08-08 — see item 6 below); `t_fire`/`a_limit` remain EDA guesses (calibration found them
  empirically inert on this corpus, see the script's grid). **GeometricContactDetector** is still
  rule-based with `delta_m` unvalidated against the full dataset.

---

## Key open items

1. **Task 3 (full-corpus retrain) — partially complete.** `scripts/train_full_corpus.py` exists
   and is tested (`tests/test_full_corpus.py`); fire/human detectors have real full-corpus
   checkpoints and results (`reports/full_corpus_eval_ready.json`, in `Full_Corpus_Engineering_Report.pdf`
   section 2). Contact detection's natural-ratio path is trained on only a ~9% stride-sampled
   subset (`scripts/train_thermox3d_subset.py`, the `ThermoX3DSubsetTraining` scheduled task) —
   the full, un-subsampled contact-training run has never completed end-to-end. **That subset
   checkpoint's chunked-training loop is also confirmed broken** (see item 7 below) — its
   `tp=0/tn=348/fp=0/fn=122` result in section 2 is invalid, not a real finding.
2. **Task 4 (50/50 balanced retrain) — DONE.** `scripts/train_balanced_corpus.py` +
   `thermal_algorithms/training/balance.py` retrain/recalibrate all six in-scope detectors (Otsu,
   FireSVM, AdaptiveThreshold, HOGSVM, MobileNetSSD, ThermoX3D — MV-STGCN excluded, see item 8)
   on a genuinely resampled 50/50 pool; checkpoints in `checkpoints_balanced/`, results in
   `Full_Corpus_Engineering_Report.pdf` section 4. ThermoX3D's balanced checkpoint specifically was retrained
   2026-08-09 after fixing the training-loop bug in item 7 — see that item for its real numbers.
3. **Task 5 (final consolidated report) — DONE, and actively maintained.** `scripts/
   generate_full_report.py` → `reports/Full_Corpus_Engineering_Report.pdf` covers both natural-ratio (§2) and
   balanced (§4) results for every in-scope detector, plus algorithm derivations
   (`reports/algorithm_derivations.md`) and a consolidated write-up of earlier investigative work
   (`reports/historical_investigations.md`) that predates this pipeline. Regenerate after any
   checkpoint/eval change — it is NOT hand-edited.
4. **Two pre-existing `test_mobilenet_ssd.py` failures** against the currently-installed
   `torch`/`opencv` versions — see the Commands section. Doesn't block training (the detector
   still fits/predicts), but the backbone shape assumption and save/load determinism should be
   revisited.
5. **Homography for contact detection** — synthetic homographies are used in all demos/training;
   real H matrices require the on-site Hot-Point Calibration procedure (`image_annotator` can
   produce one per session, but none is wired into `train_full_corpus.py` yet).
6. **Threshold calibration — DONE for `OtsuFireDetector.t_ign` (2026-08-08).**
   `scripts/calibrate_otsu_thresholds.py` grid-searches t_ign/t_fire/a_limit against ~51k frames
   sampled the same way as `FireSVMDetector`'s full-corpus training, restricted to precision>=0.8
   (an unconstrained F1-only search degenerates to predicting fire on nearly every frame — F1 never
   penalizes false positives via true negatives — rejected as operationally useless; see
   `reports/otsu_threshold_calibration.json`). New default `t_ign=41.5°C` (was `45°C`, never
   validated); held-out result on `fire_test_scenes`: 79.7% acc / 83.0% prec / 39.2% rec / 53.2% F1
   / 7.0% mean IoU, vs. the uncalibrated default's 78.4% / 100% / 26.8% / 42.3% / 4.5% — a real
   improvement (+12.3pp F1/recall for -17pp precision, still only ~3.4% false-alarm rate). `t_fire`
   (`60°C`) and `a_limit` (`≈3%` of frame) are UNCHANGED — calibration found both empirically inert
   across their whole tested range at the validated `t_ign` (every real hot blob in this corpus is
   small, <0.5% of frame, so the two-tier ignition/potential-fire split barely engages its second
   branch here) — not re-validated in the sense of "confirmed optimal", just "no data-driven reason
   to move them". `GeometricContactDetector`'s `delta_m` remains unvalidated.
7. **ThermoX3D's chunked-training loop was broken; FIXED for the balanced regime only
   (2026-08-09).** A fresh optimizer per chunk + per-chunk-recomputed normalisation stats produced
   a degenerate constant classifier — both the natural-ratio subset checkpoint (item 1) and the
   first balanced-retrain checkpoint showed the IDENTICAL `tp=0/tn=348/fp=0/fn=122` result on
   held-out data, which was originally (wrongly) reported as evidence against class imbalance
   being the cause. Fixed in `thermal_algorithms/contact_detection/thermo_x3d.py` (persistent
   AdamW, opt-in frozen normalisation via `set_normalization()`, `T` 16→5) +
   `thermal_algorithms/training/balance.py` (parent-run-safe train/val split) +
   `scripts/train_balanced_corpus.py` (early stopping, threshold recalibration) — **balanced
   regime only**; retrained and verified non-degenerate: recall 0%→100%, F1 42.1% (precision
   26.6%, still poor — a genuine data-scarcity limitation, only ~188 real positive contact frames
   exist total, not a remaining bug). The natural-ratio path (item 1's subset checkpoint) is
   **not** fixed by this — it still feeds single-class chunks. See memory/report caveats for
   `ThermoX3DDetector` in both section 2 and section 4 for the full writeup.
8. **MVSTGCNDetector excluded from further work (project-owner decision, 2026-08-08).** Confirmed
   by an earlier, independent investigation (now in `reports/historical_investigations.md` §5.2–
   5.3): a checkpoint-persistence bug (constructor-time `homography` stored outside the persisted
   `_params`, so it silently reset to `None` on load) was found and fixed once already, and even
   with that fixed, MV-STGCN remained the worst of the three contact detectors (~25-30% false-alarm
   rate) for structural reasons — it inherits the Geometric detector's homography-noise problem and
   loses its defining contact evidence to blob-merge exactly when two people touch. Not retrained
   in the balanced pass; left as a documented negative result.
