# Live Thermal Monitor

Real-time PyQt6 operator app for the Smart Thermal System. Captures three live
Waveshare/MI48 thermal cameras on a Raspberry Pi 5, runs the
`thermal_algorithms` detectors in parallel, lets the operator hot-swap the
algorithm per task at runtime, draws live boxes/alerts, shows a live bird's-eye
homography view for the geometric touch detector, and has a room-geometry
calibration mode with a 3-D model.

It runs end-to-end on any machine via recorded-session playback or a synthetic
generator, so you can develop without hardware.

## Install

```bash
pip install -e ".[live]"     # PyQt6, pyqtgraph, PyOpenGL (+ pysenxor-lite on the Pi)
```

`pysenxor-lite` is only needed on the Pi for live cameras; it is imported lazily
so playback/synthetic runs work without it.

## Run

```bash
# Dependency-free smoke run (procedural warm blobs)
python -m apps.live_monitor.main --source synthetic

# Replay a recorded session (reads chN_raw_data.npz)
python -m apps.live_monitor.main --source playback --session datasets/waveshare_work/<scene>

# Raspberry Pi with real MI48 cameras
python -m apps.live_monitor.main --source senxor --profile Waveshare_26984

# Headless self-check (build everything, process a few frames, exit)
python -m apps.live_monitor.main --smoke --source synthetic
```

Other flags: `--profile {Waveshare_26984,MLX90640}`, `--restricted`,
`--checkpoints <dir>`, `--target-fps N`.

## UI

- **Three camera panels** — thermal feed (inferno colormap), with live **person**
  (cyan) and **fire** (red) bounding boxes; hover for a per-pixel temperature.
- **Per-task selector bars** — choose the Fire / Person / Touch algorithm live.
  Options without a trained checkpoint (or missing torch/scikit-image) are greyed
  out; rule-based detectors are always available.
- **Touch banner** — turns red below the feeds whenever contact is detected.
- **Bird's-eye dock** (right) — appears only when the **Geometric** touch detector
  is selected: floor rectangle, per-camera projected foot-points, fused actors,
  a motion trail, and red contact links labelled with the floor distance.
- **Calibration → Calibrate room…** — enter room L×W×H (each capped at 5 m → max
  5×5×5 volume), per-camera pose + lens FOV, a measured floor rectangle, and
  pixel↔floor correspondences; a rotatable 3-D room model updates live. Solving
  writes `homography_calibration.npz` in the same format `image_annotator` uses
  (so the contact stack and `scripts/visualize_birdseye.py` read it unchanged).
- **Debug dock** (left) — pipeline/ camera FPS, latency and dropped-frame
  counters; live threshold sliders (`t_ign`, `t_fire`, person score, `δ`, `ε`);
  pause / step / snapshot; fixed-vs-per-frame temperature scale.
- **Flight recorder** — keeps the last ~10 s of all three cameras and dumps them
  to `recordings/clip_*.npz` on demand or automatically on any alarm.

## Architecture

```
main.py            entry point / arg parsing
capture/           FrameSource ABC + senxor (Pi MI48), playback, synthetic backends
detectors.py       per-task registry; rule-based + checkpoint-loaded options
rendering.py       Frame → RGB (inferno LUT) + QPainter overlay helpers
runner.py          Qt-free core: triplet sync, ThermalPipeline, live swap/tuning
pipeline_worker.py Qt threads: per-camera capture + pipeline thread (+ CPU tuning)
calibration.py     homography solve + annotator-compatible save/load
ui/                main_window, camera_view, birdseye_widget, calibration_dialog, room3d, debug_panel
```

The non-UI layers (`capture`, `detectors`, `rendering`, `runner`, `calibration`,
`ui/room3d`) are Qt-free and unit-tested in `tests/test_live_monitor.py`.

### Threading & CPU

One capture thread per camera writes a mutex-guarded latest-frame slot
(drop-to-latest — a fast camera never blocks on a slow one, overwrites are
counted). One pipeline thread processes the newest triplet and emits results to
the UI thread. `configure_cpu()` caps OpenCV/PyTorch thread pools so the three
detectors plus the UI don't oversubscribe the Pi 5's cores.

### The MI48 "on-board filter"

`SenxorSource` enables the camera's on-chip denoising on connect — temporal
(`FILTER_CONTROL` + `FILTER_SETTING_1`), `MEDIAN_CTRL`, `STARK_CTRL`, and
`MMS_CTRL` — so frames arrive already filtered. The exact register values live in
`Mi48FilterConfig` and should be pinned against the MI48 datasheet / on-device
SDK example.

## Calibration — what to measure for best accuracy

Geometric-touch accuracy is bounded by homography quality and metric scale.
Provide:

1. Interior room **L, W, H** (≤ 5 m each).
2. A **measured floor rectangle** (true metric W×H) with its 4 corners clicked in
   each camera — sets the metric scale.
3. **≥ 4 (ideally 6–8) hot-point correspondences per camera** at known floor
   (X, Y), spread near→far — far more robust than the rectangle alone.
4. Per-camera **mounting pose** (x, y, z, yaw, tilt) and **lens FOV (45°/90°)**.
5. The **inter-camera time offset** (~50 ms I2C phase shift) for moving subjects.
6. **Emissivity** (skin ≈ 0.98) + an ambient reference temperature so °C readings
   and fire thresholds are trustworthy.

Note: the homography is a floor-plane map assuming subjects stand on the floor
(foot-point = bbox bottom-centre at Z = 0); accuracy degrades for seated/lying
subjects.
```
