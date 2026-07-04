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

## Run — live on the Raspberry Pi (3× Waveshare Thermal Camera Module)

One-time setup (see also `thermal_algorithms/acquisition/README.md`):

```bash
sudo raspi-config                 # Interface Options → enable SPI and I2C
# /boot/firmware/config.txt: add `dtoverlay=spi0-0cs` below `dtparam=spi=on`
sudo apt install python3-venv python3-tk
python3 -m venv ~/demo-env && source ~/demo-env/bin/activate
pip install -r requirements.txt
pip install gpiozero smbus2 spidev crcmod          # acquisition extras
wget https://files.waveshare.com/wiki/Thermal_Camera_Module/Thermal_Camera_Hat.zip
unzip Thermal_Camera_Hat.zip && pip install -e pysenxor-master/
```

Adjust the wiring in `demo/cams_pi.json` (I2C address / SPI device /
CS / DATA_READY / RESET pins per camera — camera 0 matches the wiki
default), then:

```bash
python -m demo.app --live --config demo/cams_pi.json
```

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
