"""Preprocessing — Tateno pipeline (Engineering Report § 4.4.1).

Spatial Denoising (Gaussian) → Background Subtraction (mean-field environmental
filtering) → Residual Rectification (L1 magnitude).
"""

from thermal_algorithms.preprocessing.base import Preprocessor
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline

__all__ = ["Preprocessor", "TatenoPipeline"]
