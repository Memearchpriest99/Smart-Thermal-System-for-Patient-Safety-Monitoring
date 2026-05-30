# Smart Thermal System for Patient Safety Monitoring

A privacy-preserving edge system for closed psychiatric wards, developed in collaboration with the **Center for Mental Health in Be'er Sheva**.

The system uses low-resolution thermal sensors — no optical cameras — to detect four safety-critical scenarios in real time, without capturing personally identifiable visual information.

| Scenario | Detection method |
|---|---|
| 🔥 Fire ignition (lighters, cigarettes) | Per-camera fire detection |
| ⚡ Violent acts / inappropriate contact | 3-view contact detection |
| 🚫 Restricted-area breach | Human detection |
| 👁 General presence monitoring | Human detection |

**Hardware:** Raspberry Pi 5 · 3× MLX90640 32×24 thermal sensors (or Waveshare 80×62) · TCA9548A I2C multiplexer  
**Target cost:** ~$250/unit  
**Authors:** Guy Chen · Yaniv Blau · Roy Lieberman  
**Supervisor:** Or Zilberberg · **Advisor:** Dr. Oshrit Hoffer  
**Institution:** Afeka Academic College of Engineering in Tel-Aviv

---

## Quick start

```bash
pip install -e ".[dev]"

# Run the end-to-end pipeline demo (falls back to synthetic data if no dataset)
python examples/demo_pipeline.py

# Reproduce the fire detection evaluation table (§ 5.3.4)
# python scripts/eval_fire_detection.py   ← available once dataset is labeled

# Run tests
python -m pytest tests/test_pipeline.py tests/test_otsu_pipeline.py -v
```

---

## Repository structure

```
thermal_algorithms/     Python package — all algorithms and training infrastructure
  core/                 Base classes, data types, sensor profiles, checkpoints
  preprocessing/        Tateno preprocessing pipeline
  human_detection/      3 detector alternatives (§ 4.4.2)
  fire_detection/       2 detector alternatives (§ 4.4.4)
  contact_detection/    3 detector alternatives + multi-view utilities (§ 4.4.3)
  training/             Datasets, metrics, evaluation harness
  pipeline.py           Runtime integration — ThermalPipeline

examples/               Runnable demo scripts (all work with synthetic data)
tests/                  363+ unit and integration tests
outputs/                Generated figures and calibration files (git-ignored)
```

See [`CLAUDE.md`](CLAUDE.md) for developer-facing architecture notes, test commands, and open items.

---

## System flow

```
Thermal sensors (3 cameras)
        │
        ▼
  ThermalPipeline.process(frame0, frame1, frame2)
        │
  restricted=True ──► Human detection only ──► RESTRICTED_AREA alert
        │
  restricted=False
        ├──► Preprocessing (Tateno) ──► Human detection
        ├──► Fire detection (per camera)  ──► FIRE alert
        └──► Contact detection (3-view)   ──► CONTACT alert
```

---

## Dataset annotation convention

YOLO-format bounding boxes, one `.txt` per frame, alongside `.npz` raw sensor data:

```
class 0 = fire / ignition source
class 1 = person
```

Contact labels (no bounding box representation at 32×24) are stored in per-session `contact_labels.csv` files. See [`thermal_algorithms/training/README.md`](thermal_algorithms/training/README.md).

---

## License

Proprietary — Afeka Academic College of Engineering. All rights reserved.
