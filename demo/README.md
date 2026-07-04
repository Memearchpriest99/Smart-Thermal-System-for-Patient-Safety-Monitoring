# Smart Thermal System — Live Demo

Real-time 3-camera thermal monitor with person / fire bounding-box overlays
and contact (touch) detection. Torch-free at runtime:

| Task | Model | Runtime |
|---|---|---|
| Person detection | MobileNet-SSD (`Waveshare_26984_raw.onnx`) | onnxruntime, numpy decode |
| Fire detection | FireSVM (`fire_svm_detector/_default.thalg`) | scikit-learn |
| Contact detection | Thermo-X3D T5v2 (`Waveshare_26984_T5_v2.ftz.onnx`, denormal-flushed) | onnxruntime |

Preprocessing is the session-local rolling 25th-percentile background →
Tateno residual (no per-room calibration, ~2 s warm-up). Every algorithm runs
on its own worker thread; queues drop stale frames so the UI never lags the
sensor.

## Run — replay (any machine)

```bash
python -m demo.app --replay demo/sample_session        # bundled fallback clip
python -m demo.app --replay <session_dir> [<dir2> ...] # cycles sessions
```

A session folder needs `ch0/1/2_raw_data.npz` or `ch0/1/2_thermal.npz`
(key `frames`, shape `(N, 62, 80)` float32 °C).

## Run — live on the Raspberry Pi (3× Waveshare module over USB-C)

The deployed wiring: each camera's USB-C port to a Pi USB-A socket. The
modules enumerate as serial devices (VID 0x0416) — no GPIO, no
raspi-config, no SPI/I2C setup.

One-time setup:

```bash
sudo apt install python3-venv python3-tk
python3 -m venv ~/demo-env && source ~/demo-env/bin/activate
pip install -r requirements.txt pyserial crcmod
wget https://files.waveshare.com/wiki/Thermal_Camera_Module/Thermal_Camera_Hat.zip
unzip Thermal_Camera_Hat.zip && pip install -e pysenxor-master/
sudo usermod -aG dialout $USER && newgrp dialout   # serial-port permission
```

Then simply:

```bash
python -m demo.app --live                 # auto-detects the 3 cameras
```

Camera IDs follow **physical USB-socket order** (stable across reboots).
Verify once at rehearsal which socket is camera 0/1/2 — wave a hand in
front of each; to reorder, swap cables or pass explicit ports:

```bash
python -m demo.app --live --ports /dev/ttyACM0 /dev/ttyACM2 /dev/ttyACM1
```

(Alternative SPI/I2C HAT wiring is still supported via
`--spi-config demo/cams_pi.json` — see `thermal_algorithms/acquisition/README.md`.)

**Crowd-demo insurance:** if a camera acts up mid-demo, switch to
`--replay demo/sample_session` — same UI, recorded data.

## Offscreen self-test (no display needed)

```bash
python -m demo.capture demo/sample_session --wait-alarm --after 90 --out /tmp/demo.png
```

Prints the contact confidence it captured; the PNG shows the full UI.

## Known behaviors

- "calibrating background" for the first ~2 s: the rolling-p25 background
  needs 16 frames before residuals (and contact detection) start.
- The contact model was trained at 8 Hz — keep `--fps 8` unless retrained.
- Alarm threshold 0.35 with 3-frame persistence
  (`thermo_x3d_detector/Waveshare_26984_T5_v2.meta.json`).
