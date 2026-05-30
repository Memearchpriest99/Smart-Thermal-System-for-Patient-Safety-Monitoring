"""Internal multi-view utilities used by the three contact detection algorithms.

  - homography.py — Homography calibration (Hot-Point Calibration) and projection
  - tracker.py    — Kalman filter (Constant Velocity model) for I2C phase-shift correction
  - fusion.py     — Foot-point projection + clustering to unique Actor nodes
"""

from thermal_algorithms.contact_detection.multi_view.homography import (
    solve_homography_from_markers,
    project_foot_point,
)
from thermal_algorithms.contact_detection.multi_view.tracker import (
    KalmanTrack,
    PerCameraTracker,
)
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections

__all__ = [
    "solve_homography_from_markers",
    "project_foot_point",
    "KalmanTrack",
    "PerCameraTracker",
    "fuse_detections",
]
