# Smart Thermal System — Demo (Linux / Raspberry Pi)

Privacy-preserving patient-safety monitoring on thermal video only —
**no optical cameras**. This branch is the self-contained live demo:
three 80×62 thermal feeds with real-time person and fire detection
overlays, and contact (touch) detection on a strip at the bottom.

**Model weights are included** (`checkpoints/`, ~2 MB) — clone and run.

| Task | Model | Runtime |
|---|---|---|
| Person detection | MobileNet-SSD | onnxruntime |
| Fire detection | FireSVM (Otsu features + RBF SVM) | scikit-learn |
| Contact detection | Thermo-X3D T5v2, denormal-flushed ONNX | onnxruntime |

Each camera has its own worker thread (background model + person + fire);
contact runs on a fourth thread over the 3-view residual window. Queues
drop stale frames, so the UI always shows the present.

## Quick start (replay — no hardware needed)

```bash
sudo apt install python3-venv python3-tk
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python -m demo.app --replay demo/sample_session
```

`demo/sample_session` is a bundled 44 s three-camera recording that
triggers person boxes and a contact alarm — it is also the on-stage
fallback if a camera misbehaves.

## Live mode (3× Waveshare Thermal Camera Module on a Raspberry Pi 5)

One-time Pi setup:

```bash
sudo raspi-config        # Interface Options → enable SPI and I2C
# /boot/firmware/config.txt: add `dtoverlay=spi0-0cs` below `dtparam=spi=on`
pip install gpiozero smbus2 spidev crcmod
wget https://files.waveshare.com/wiki/Thermal_Camera_Module/Thermal_Camera_Hat.zip
unzip Thermal_Camera_Hat.zip && pip install -e pysenxor-master/
```

Match `demo/cams_pi.json` to your wiring (camera 0 is the wiki-default
HAT wiring; I2C address is selected by the on-board 0R resistor), then:

```bash
python -m demo.app --live --config demo/cams_pi.json
```

## Self-test without a display

```bash
python -m demo.capture demo/sample_session --wait-alarm --after 90 --out demo_check.png
```

Expected output: `alarmed True` with contact confidence ≈ 1.0, and a PNG
of the full UI.

See `demo/README.md` for details and known behaviors (2 s background
warm-up, 8 Hz requirement, alarm threshold).

---
Afeka College of Engineering — Guy Chen, Yaniv Blau, Roy Lieberman.
Supervisor: Or Zilberberg · Advisor: Dr. Oshrit Hoffer.
