"""Preprocessor — abstract base for the pre-processing pipeline (§ 4.4.1)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

import numpy as np

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.types import Frame


class Preprocessor(ThermalAlgorithm):
    """Transforms a raw thermal frame into a clean, rectified residual frame.

    Contract
    --------
    Input  : a single `Frame` (single-view; multi-view callers apply this
             independently per camera).
    Output : a single `Frame` of the same resolution, same timestamp,
             same camera_id — `data` is the rectified residual.

    Calibration
    -----------
    `fit(calibration_frames)` consumes K frames captured with no targets
    present (an empty room) and computes the background model
        B(x, y) = (1/K) Σ_k I_calib(x, y, k)
    described in § 4.4.1 step 2 (Background Subtraction).

    Preprocessors are not "trainable" in the ML sense — `is_trainable=False` —
    but `fit()` is still required (calibration), so `predict()` raises if
    called before `fit()`.

    Resolution behavior
    -------------------
    'parameterized' — Gaussian sigma, kernel size, and the shape of the
    background frame B(x, y) all scale with the SensorProfile.
    """

    is_trainable = False
    resolution_behavior = "parameterized"

    # ---- API --------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: Iterable[Frame] | np.ndarray,
        y: None = None,
    ) -> "Preprocessor":
        """Learn the background reference B(x, y) from calibration frames.

        Args:
            X: An iterable of `Frame`s, or a stack of raw arrays of shape
               (K, H, W). All frames must match the configured SensorProfile.
            y: Unused (preprocessing is unsupervised).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Frame) -> Frame:
        """Apply the pipeline to a single frame.

        Returns a new `Frame` whose `data` is the rectified residual; metadata
        (timestamp, camera_id) is preserved unchanged.
        """
        raise NotImplementedError

    # ---- Convenience alias ----------------------------------------------

    def transform(self, frame: Frame) -> Frame:
        """Alias for `predict()` — preprocessing reads more naturally as
        'transform', and matches the sklearn Transformer convention."""
        return self.predict(frame)

    def fit_transform(self, X: Iterable[Frame] | np.ndarray) -> list[Frame]:
        """Fit on `X` then transform every input frame in it.

        Useful for one-shot calibration where the calibration frames double
        as the first batch to process.
        """
        self.fit(X)
        if isinstance(X, np.ndarray):
            raise TypeError(
                "fit_transform on raw ndarray is ambiguous (no timestamps). "
                "Pass an iterable of Frame objects instead."
            )
        return [self.predict(f) for f in X]
