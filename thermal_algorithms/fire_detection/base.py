"""FireDetector — abstract base for the two fire detection algorithms
(§ 4.4.4)."""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.types import Frame, FireAlert


class FireDetector(ThermalAlgorithm):
    """Detects thermal hazards (active fires, ignition sources) in a thermal frame.

    Contract
    --------
    Input  : a single `Frame` (typically pre-processed; the detector itself
             does not apply the § 4.4.1 pipeline).
    Output : a single `FireAlert`.

    Statefulness
    ------------
    The pipelined approach (§ 4.4.4 step 3 — Temporal Mass Gradient) maintains
    a per-tracked-object growth counter S_n and a sliding history buffer of
    blob areas A_n over the last T_measure seconds. Callers do not need to
    manage this; `reset()` clears it.

    The SVM alternative is stateless.

    Training / calibration
    ----------------------
    The pipelined approach is rule-based: `fit()` is at most a calibration
    step that tunes the decision-tree thresholds (T_ign, T_fire, A_limit)
    from a labelled calibration set, or simply returns self if the
    hand-engineered defaults are used.

    The SVM alternative is trainable: `fit(X, y)` consumes labelled frames
    with `y` in {SAFE, ACTIVE_COMBUSTION}.

    Resolution behavior
    -------------------
    'parameterized' for the Otsu pipeline (kernel sizes + A_limit scale with
    resolution), 'invariant' for the SVM (features are scalars).
    """

    is_trainable = False  # overridden in FireSVMDetector

    # ---- API --------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: Iterable[Frame],
        y: Iterable[FireAlert] | None = None,
    ) -> "FireDetector":
        """Train (SVM) or calibrate (Otsu pipeline thresholds).

        Args:
            X: Iterable of training/calibration frames.
            y: Parallel iterable of ground-truth alerts. None for
               non-trainable variants.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Frame) -> FireAlert:
        """Classify a single frame.

        For the pipelined approach, `predict()` is stateful — calling it
        repeatedly on consecutive frames updates the temporal mass-gradient
        counter S_n internally. Call `reset()` between independent sequences.

        For the SVM approach, `predict()` is stateless.

        Returns:
            A `FireAlert` whose `timestamp` equals `X.timestamp`.
        """
        raise NotImplementedError
