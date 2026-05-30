"""ContactDetector — abstract base for the three contact detection algorithms
(§ 4.4.3).

All three alternatives consume 3-view input but differ in what they need from
each view:
    * Geometric (§ 4.4.3.1)  — needs detections per view + homography
    * MV-STGCN  (§ 4.4.3.2)  — needs detections per view (which it can compute
                               internally with its embedded YOLO-Nano) +
                               homography + temporal history
    * Thermo-X3D (§ 4.4.3.3) — needs raw frames per view; pixel-based, no
                               detections required
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable, Optional

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.types import (
    Frame,
    Detection,
    ContactEvent,
    HomographyMatrices,
)

# A "3-view" is always a 3-tuple of Frames (one per camera).
ThreeViewFrames = tuple[Frame, Frame, Frame]
ThreeViewDetections = tuple[list[Detection], list[Detection], list[Detection]]


class ContactDetector(ThermalAlgorithm):
    """Detects physical contact between subjects across three thermal cameras.

    Contract
    --------
    Input  : a 3-tuple of `Frame`s (one per camera, synchronized as well as
             the Kalman tracker can manage given the I2C phase shift).
             Optionally, a 3-tuple of pre-computed `Detection` lists from an
             upstream HumanDetector — saves duplicate work when the same
             detections feed both the human-presence and contact pipelines.
    Output : a single `ContactEvent`.

    Statefulness
    ------------
    MV-STGCN and Thermo-X3D both consume a sliding window of past frames
    (16 frames ≈ 2 seconds at 8 Hz). The detector owns this buffer
    internally; callers do not need to manage it. `reset()` clears the buffer
    and resets the tracker.

    Calibration
    -----------
    All three alternatives need homography matrices (H1, H2, H3) to project
    image-plane detections to the ground plane. Two construction modes:

        1. Caller provides them at __init__ via `homography=...`.
        2. Caller calls `calibrate_homography(marker_correspondences)` to
           solve them via SVD from heated-target calibration points.

    Training
    --------
    Trainable detectors (MV-STGCN, Thermo-X3D) consume labelled triplets in
    `fit()`. Geometric needs only homography calibration.
    """

    is_trainable = False  # overridden in MVSTGCNDetector, ThermoX3DDetector

    # ---- Construction overload -----------------------------------------

    def __init__(
        self,
        *,
        homography: Optional[HomographyMatrices] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._homography: Optional[HomographyMatrices] = homography

    @property
    def homography(self) -> Optional[HomographyMatrices]:
        return self._homography

    def set_homography(self, H: HomographyMatrices) -> "ContactDetector":
        """Inject pre-calibrated homography matrices (chainable)."""
        self._homography = H
        return self

    # ---- API --------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: Iterable[ThreeViewFrames],
        y: Iterable[ContactEvent] | None = None,
    ) -> "ContactDetector":
        """Train (for MV-STGCN/Thermo-X3D) or noop (for Geometric).

        Args:
            X: Iterable of 3-view frame triplets. For temporal detectors,
               each element of X represents a single timestep — the detector
               itself maintains the sliding window across consecutive calls.
            y: Parallel iterable of ground-truth `ContactEvent`s.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        X: ThreeViewFrames,
        detections: Optional[ThreeViewDetections] = None,
    ) -> ContactEvent:
        """Infer contact for one timestep.

        Args:
            X: The current frame from each camera.
            detections: Optional pre-computed detections per view. If None,
                the detector either computes them internally (MV-STGCN's
                embedded YOLO) or doesn't need them at all (Thermo-X3D
                operates on raw pixels).

        Returns:
            A `ContactEvent`. The `timestamp` should equal the maximum
            frame timestamp in `X` (typically the camera-3 timestamp, which
            is read last on the I2C bus).
        """
        raise NotImplementedError

    # ---- Homography calibration ----------------------------------------

    def calibrate_homography(
        self,
        marker_correspondences: list[
            tuple[int, list[tuple[tuple[float, float], tuple[float, float]]]]
        ],
    ) -> HomographyMatrices:
        """Solve for H1, H2, H3 from heated-target marker correspondences.

        Concrete implementations live in `contact_detection/multi_view/
        homography.py`. This method is a thin wrapper that calls into that
        utility and stores the result on `self`.

        Args:
            marker_correspondences: For each camera, a list of
                ((u, v), (X_w, Y_w)) pairs — image-plane pixel ↔ world-plane
                meters. At least 4 pairs per camera.

        Returns:
            The fitted `HomographyMatrices` (also stored on `self`).
        """
        from thermal_algorithms.contact_detection.multi_view.homography import (
            solve_homography_from_markers,
        )

        H = solve_homography_from_markers(marker_correspondences)
        self._homography = H
        return H
