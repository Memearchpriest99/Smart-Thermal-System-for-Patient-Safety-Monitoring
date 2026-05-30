"""Human Detection - three alternative algorithms (Section 4.4.2).

  1. Adaptive Gaussian Thresholding (classical CV)  - AdaptiveThresholdDetector
  2. HOG + Linear SVM (feature-based ML)            - HOGSVMDetector
  3. MobileNet-SSD (deep learning)                  - MobileNetSSDDetector

HOGSVMDetector and MobileNetSSDDetector are imported lazily so that
AdaptiveThresholdDetector (no heavy deps) remains importable on machines
where scikit-image or PyTorch are not installed.

Import them directly or via this package:

    from thermal_algorithms.human_detection import AdaptiveThresholdDetector
    from thermal_algorithms.human_detection import HOGSVMDetector        # needs scikit-image
    from thermal_algorithms.human_detection import MobileNetSSDDetector  # needs torch
"""

from thermal_algorithms.human_detection.base import HumanDetector
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector


def __getattr__(name: str):
    """Lazy imports for heavy-dependency detectors."""
    if name == "HOGSVMDetector":
        from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
        return HOGSVMDetector
    if name == "MobileNetSSDDetector":
        from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
        return MobileNetSSDDetector
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "HumanDetector",
    "AdaptiveThresholdDetector",
    "HOGSVMDetector",
    "MobileNetSSDDetector",
]
