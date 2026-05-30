# human_detection

Three alternative human detection algorithms (§ 4.4.2), all implementing `HumanDetector` → `predict(frame) → list[Detection]`.

## Algorithms

### AdaptiveThresholdDetector (§ 4.4.2.1)
Classical CV. No training required.
- Local Gaussian-weighted mean thresholding (`D(x,y) = 1 if I > local_mean + C`)
- Morphological closing to fill thermal gaps within blobs
- Geometric filtering: area, solidity, aspect ratio gates

```python
from thermal_algorithms.human_detection import AdaptiveThresholdDetector
detector = AdaptiveThresholdDetector(MLX90640, c_offset=0.5).fit([])
dets = detector.predict(preprocessed_frame)
```

### HOGSVMDetector (§ 4.4.2.2)
HOG feature extraction + LinearSVC sliding-window detector. Requires `scikit-image`.
- Canonical window size derived from physical person dimensions (90 cm × 45 cm at 2 m)
- NMS post-processing; negative sampling built into `fit()`

### MobileNetSSDDetector (§ 4.4.2.3)
Micro MobileNet-SSD with per-profile architecture configs. Requires `torch`.
- Complete IoU (CIoU) loss for precise localization at 32×24
- Two separate checkpoints: one for MLX90640, one for Waveshare
- Lazy import: `from thermal_algorithms.human_detection import MobileNetSSDDetector`

## Import note

`HOGSVMDetector` and `MobileNetSSDDetector` are lazy-loaded so `AdaptiveThresholdDetector` remains importable without heavy dependencies:

```python
from thermal_algorithms.human_detection import AdaptiveThresholdDetector  # always works
from thermal_algorithms.human_detection import HOGSVMDetector              # needs scikit-image
from thermal_algorithms.human_detection import MobileNetSSDDetector        # needs torch
```

## YOLO class convention

```
class 0 = fire   (FIRE_CLASS_ID)
class 1 = person (PERSON_CLASS_ID)
```
