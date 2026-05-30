"""Fire Detection — two alternative algorithms (Engineering Report § 4.4.4).

  1. Otsu Pipeline (Adaptive Segmentation → Decision Tree → Temporal Mass Gradient)
  2. Feature-based SVM on statistical descriptors
"""

from thermal_algorithms.fire_detection.base import FireDetector
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector

__all__ = ["FireDetector", "OtsuFireDetector", "FireSVMDetector"]
