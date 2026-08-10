"""Contact Detection — four alternative algorithms (Engineering Report § 4.4.3).

  1. Classical Geometric Fusion Pipeline
  2. Multi-View Spatiotemporal Graph CNN (MV-STGCN)
  3. Volumetric Thermal Interaction Network (Thermo-X3D)
  4. Rule-Based Temporal Contact Tracker (RBTCT) — homography-free and
     training-free; see rbtct.py and algorithm_derivations.md §8.

All three consume 3-view input. Shared multi-view utilities (homography, Kalman
tracker, foot-point fusion) live in the multi_view/ sub-package.
"""

from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewFrames,
    ThreeViewDetections,
)
from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
from thermal_algorithms.contact_detection.rbtct import RBTCTDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector

__all__ = [
    "ContactDetector",
    "ThreeViewFrames",
    "ThreeViewDetections",
    "GeometricContactDetector",
    "MVSTGCNDetector",
    "RBTCTDetector",
    "ThermoX3DDetector",
]
