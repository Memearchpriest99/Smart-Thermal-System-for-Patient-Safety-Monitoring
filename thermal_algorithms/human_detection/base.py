"""HumanDetector — abstract base for the three human detection algorithms
(§ 4.4.2)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.types import Frame, Detection


class HumanDetector(ThermalAlgorithm):
    """Detects human presence in a single thermal frame.

    Contract
    --------
    Input  : a single `Frame` (typically pre-processed; the detector itself
             does not apply the § 4.4.1 pipeline).
    Output : list of `Detection`s. Empty list means no human detected.
             Each Detection's `camera_id` is set from the input frame.

    Training / calibration
    ----------------------
    Trainable detectors (HOG+SVM, MobileNet-SSD) consume `(frames, labels)`
    in `fit()`, where `labels` is a parallel iterable of `list[Detection]`
    (ground-truth bounding boxes).

    Non-trainable detectors (Adaptive Threshold) treat `fit()` as a noop and
    return self.

    Resolution behavior
    -------------------
    Per-subclass — Adaptive Threshold is 'parameterized', HOG+SVM is
    'parameterized' (feature dim differs per profile → separate SVM weights),
    MobileNet-SSD is 'fixed' (one checkpoint per profile).
    """

    is_trainable = False  # overridden in HOGSVMDetector, MobileNetSSDDetector

    # ---- API --------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: Iterable[Frame],
        y: Iterable[list[Detection]] | None = None,
    ) -> "HumanDetector":
        """Train or calibrate the detector.

        Args:
            X: Iterable of input frames.
            y: Parallel iterable of ground-truth detections. None for
               unsupervised / non-trainable detectors.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Frame) -> list[Detection]:
        """Detect humans in a single frame.

        Returns:
            A list of `Detection` instances. Empty list indicates 'no human
            detected'. The detector should attach `camera_id` from `X` to
            every returned Detection.
        """
        raise NotImplementedError
