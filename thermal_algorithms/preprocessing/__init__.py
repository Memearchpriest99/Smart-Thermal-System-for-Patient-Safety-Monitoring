"""Preprocessing — Tateno pipeline (Engineering Report § 4.4.1) and the
GlobalNorm baseline (Task 2 comparison).

Spatial Denoising (Gaussian) → Background Subtraction (mean-field environmental
filtering, TatenoPipeline / per-frame global mean, GlobalNormPreprocessor) →
Residual Rectification (L1 magnitude).
"""

from thermal_algorithms.preprocessing.base import Preprocessor
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline

__all__ = ["Preprocessor", "TatenoPipeline", "GlobalNormPreprocessor"]
