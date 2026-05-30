# thermal_algorithms

Python package implementing all algorithms from Chapter 4.4 of the Engineering Report, behind a uniform interface so implementations are interchangeable at the call site.

## Universal interface

Every algorithm inherits `ThermalAlgorithm` and exposes:

```python
algo = ConcreteAlgorithm(sensor_profile=MLX90640, **hyperparams)
algo.fit(X_train, y_train)        # training or calibration
y_pred = algo.predict(X_test)     # inference
algo.save("checkpoint.thalg")
same = ConcreteAlgorithm.load("checkpoint.thalg")
algo.reset()                      # clear temporal state (stateful detectors)
```

## Data types

All algorithms share frozen dataclasses from `core/types.py`:

| Type | Used by |
|---|---|
| `Frame(data, timestamp, camera_id)` | Everything |
| `Detection(bbox, score, class_id)` | Human detectors → contact detectors |
| `FireAlert(level, timestamp, blob_features)` | Fire detectors |
| `ContactEvent(actors, pairs_in_contact, timestamp)` | Contact detectors |
| `HomographyMatrices(h1, h2, h3)` | Contact detectors |

## Module map

| Module | What it implements |
|---|---|
| [`core/`](core/README.md) | Base ABC, types, sensor profiles, checkpoint registry |
| [`preprocessing/`](preprocessing/README.md) | Tateno pipeline (§ 4.4.1) |
| [`human_detection/`](human_detection/README.md) | Adaptive threshold, HOG+SVM, MobileNet-SSD (§ 4.4.2) |
| [`fire_detection/`](fire_detection/README.md) | Otsu pipeline, Fire SVM (§ 4.4.4) |
| [`contact_detection/`](contact_detection/README.md) | Geometric fusion, MV-STGCN, Thermo-X3D (§ 4.4.3) |
| [`training/`](training/README.md) | Datasets, metrics, evaluation harness |
| `pipeline.py` | `ThermalPipeline` — runtime integration |

## Sensor profiles

```python
from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
```

Two pre-defined profiles. Algorithms tagged `resolution_behavior="parameterized"` auto-scale kernel sizes based on `profile.physical_pixel_size_m(distance_m)`. Algorithms tagged `"fixed"` need a separate checkpoint per profile.
