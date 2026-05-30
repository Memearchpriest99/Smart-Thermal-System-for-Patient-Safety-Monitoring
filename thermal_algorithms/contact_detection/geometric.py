"""GeometricContactDetector — classical multi-view fusion pipeline (§ 4.4.3.1).

Algorithm Flow
--------------
For each time step, given pre-computed detections from all three cameras:

1. Segmentation & Foot-Point Extraction
   The foot-point P_foot = (x + w/2, y + h) of each bounding box
   approximates the subject's contact point with the floor.

2. Ground-Plane Projection
   Apply homography H_k to each P_foot to obtain 2-D world coordinates.

3. Data Association & Clustering
   Validate cross-camera projections (discard single-source; outlier removal
   for N = 3) and cluster within ε_m metres → unique ActorPosition nodes.

4. Contact Logic
   Compute pairwise Euclidean distances on the floor plane.
   Flag pair (i, j) as contact if D_{i,j} < δ metres.

Parameters
----------
epsilon_m : clustering / cross-camera consistency threshold (default 0.5 m)
delta_m   : contact proximity threshold (default 0.5 m, per § 4.4.3)

Both are empirically calibrated for the physical width of subjects and the
intrinsic noise of the thermal projection.

Resolution behavior: 'invariant' — all computation is in world-space metres,
independent of sensor resolution.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import (
    ActorPosition,
    ContactEvent,
    Detection,
    HomographyMatrices,
)
from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewDetections,
    ThreeViewFrames,
)
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections


class GeometricContactDetector(ContactDetector):
    """Deterministic, explainable contact detector based on planar homography.

    This is the classical baseline (§ 4.4.3.1).  It requires:
      * Pre-calibrated homography matrices (via ``calibrate_homography()`` or
        injected at construction time).
      * Pre-computed ``HumanDetector`` outputs passed as ``detections`` in
        every ``predict()`` call.

    The detector has no learned parameters; ``fit()`` simply marks it as
    ready and optionally accepts a calibration set to do nothing with
    (kept for API uniformity).
    """

    name = "geometric_contact_detector"
    is_trainable = False
    resolution_behavior = "invariant"

    def __init__(
        self,
        sensor_profile: Optional[SensorProfile] = None,
        *,
        homography: Optional[HomographyMatrices] = None,
        epsilon_m: float = 0.5,
        delta_m: float = 0.5,
    ) -> None:
        """
        Args:
            sensor_profile: Optional; accepted for uniform construction but
                ignored (algorithm is resolution-invariant).
            homography: Pre-calibrated ``HomographyMatrices``.  Can also be
                set later via ``set_homography()`` or
                ``calibrate_homography()``.
            epsilon_m: Cross-camera clustering threshold in metres.
            delta_m: Contact proximity threshold in metres.
        """
        super().__init__(
            sensor_profile=sensor_profile,
            homography=homography,
            epsilon_m=epsilon_m,
            delta_m=delta_m,
        )
        self._epsilon_m = float(epsilon_m)
        self._delta_m = float(delta_m)
        self._next_track_id: int = 0

    # ---- Fit ----------------------------------------------------------------

    def fit(
        self,
        X: Iterable[ThreeViewFrames],
        y: Iterable[ContactEvent] | None = None,
    ) -> "GeometricContactDetector":
        """No-op calibration — thresholds are set at construction time."""
        self._is_fitted = True
        return self

    # ---- Predict ------------------------------------------------------------

    def predict(
        self,
        X: ThreeViewFrames,
        detections: Optional[ThreeViewDetections] = None,
    ) -> ContactEvent:
        """Detect contact for one timestep.

        Args:
            X: Current frames from each camera (used only for timestamps).
            detections: Required — pre-computed ``list[Detection]`` per
                camera from an upstream ``HumanDetector``.

        Raises:
            RuntimeError: If homography matrices have not been calibrated.
            ValueError: If ``detections`` is None.
        """
        if self._homography is None:
            raise RuntimeError(
                "GeometricContactDetector.predict() called without homography. "
                "Call calibrate_homography() or set_homography() first."
            )
        if detections is None:
            raise ValueError(
                "GeometricContactDetector requires pre-computed detections. "
                "Pass them as predict(X, detections=(dets0, dets1, dets2))."
            )

        timestamp = max(f.timestamp for f in X)

        # Project, validate, cluster → unique actor positions
        actors, self._next_track_id = fuse_detections(
            list(detections),
            self._homography,
            epsilon_m=self._epsilon_m,
            next_track_id=self._next_track_id,
        )

        # Pairwise contact check
        pairs: list[tuple[int, int]] = []
        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                d = _euclidean(actors[i].world_xy, actors[j].world_xy)
                if d < self._delta_m:
                    pairs.append((i, j))

        return ContactEvent(
            actors=tuple(actors),
            pairs_in_contact=tuple(pairs),
            timestamp=timestamp,
            confidence=1.0,
            debug={
                "n_actors": len(actors),
                "delta_m": self._delta_m,
                "epsilon_m": self._epsilon_m,
            },
        )

    def reset(self) -> None:
        """Reset the per-frame track-id counter."""
        self._next_track_id = 0

    # ---- Persistence --------------------------------------------------------

    def _state_dict(self) -> dict:
        h = self._homography
        if h is None:
            return {}
        return {
            "h1": h.h1,
            "h2": h.h2,
            "h3": h.h3,
        }

    def _load_state_dict(self, state: dict) -> None:
        import numpy as np
        if state:
            self._homography = HomographyMatrices(
                h1=np.asarray(state["h1"], dtype=np.float64),
                h2=np.asarray(state["h2"], dtype=np.float64),
                h3=np.asarray(state["h3"], dtype=np.float64),
            )
        self._is_fitted = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
