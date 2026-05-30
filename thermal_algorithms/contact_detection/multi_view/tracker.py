"""Kalman Filter tracker for correcting the TCA9548A I2C phase shift (§ 4.4.3).

Hardware context
----------------
The TCA9548A multiplexer reads the three sensors sequentially.  Camera 3 is
read approximately 40–50 ms after Camera 1.  At typical walking speeds,
subjects will have shifted position between reads, creating "ghosting" in
multi-view fusion.  The tracker solves this by forward-predicting Camera-1
detections by Δt (≈ 50 ms) so they are temporally aligned with Camera-3.

It also acts as a low-pass filter on 1-pixel quantisation jitter (the
measurement-noise covariance R absorbs ±1 px uncertainty), producing smooth
velocity vectors that are subsequently used as node features in the GCN.

Model
-----
Constant Velocity (CV) model.  State vector s_t:

    s_t = [u, v, u̇, v̇]^T

Transition matrix F (for time step Δt):

    F = [[1, 0, Δt, 0 ],
         [0, 1, 0,  Δt],
         [0, 0, 1,  0 ],
         [0, 0, 0,  1 ]]

Observation matrix H_obs (we observe the centroid pixel only):

    H_obs = [[1, 0, 0, 0],
             [0, 1, 0, 0]]

Predict:  s_{t|t-1} = F s_{t-1}
          P_{t|t-1} = F P_{t-1} F^T + Q

Update:   K = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}
          s_t = s_{t|t-1} + K (z_t - H s_{t|t-1})
          P_t = (I - K H) P_{t|t-1}

Association
-----------
PerCameraTracker uses greedy nearest-neighbour matching (adequate for ≤ 5
subjects).  Tracks that miss more than ``max_age`` consecutive frames are
pruned.  New detections that do not match any track create fresh tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Single-track Kalman filter
# ---------------------------------------------------------------------------

_H_OBS = np.array([[1., 0., 0., 0.],
                   [0., 1., 0., 0.]], dtype=np.float64)   # observation matrix

_I4 = np.eye(4, dtype=np.float64)


@dataclass
class KalmanTrack:
    """One Kalman-filter track for a single person in a single camera view.

    Attributes:
        track_id: Stable identifier assigned at birth.
        state: Current state estimate [u, v, u̇, v̇].
        cov: 4×4 state-covariance matrix P.
        age: Total number of frames this track has been alive.
        misses: Consecutive frames with no associated detection.
    """

    track_id: int
    state: np.ndarray          # (4,) float64
    cov: np.ndarray            # (4, 4) float64
    age: int = 0
    misses: int = 0

    # ---- Read-only properties ------------------------------------------------

    @property
    def position(self) -> tuple[float, float]:
        """Current estimated pixel position (u, v)."""
        return (float(self.state[0]), float(self.state[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated pixel velocity (u̇, v̇) in pixels per second."""
        return (float(self.state[2]), float(self.state[3]))

    # ---- Kalman operations --------------------------------------------------

    def predict(self, dt: float, Q: np.ndarray) -> None:
        """Advance the filter by dt seconds (in-place)."""
        F = _make_transition(dt)
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        self.age += 1

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        """Fuse a new measurement z = [u_meas, v_meas] (in-place)."""
        S = _H_OBS @ self.cov @ _H_OBS.T + R
        K = self.cov @ _H_OBS.T @ np.linalg.inv(S)
        innovation = z - _H_OBS @ self.state
        self.state = self.state + K @ innovation
        self.cov = (_I4 - K @ _H_OBS) @ self.cov
        self.misses = 0


# ---------------------------------------------------------------------------
# Per-camera multi-track manager
# ---------------------------------------------------------------------------

class PerCameraTracker:
    """Manages all Kalman tracks for a single camera view.

    Args:
        dt: Nominal time step between frames in seconds.
        sigma_process: Process noise standard deviation in pixels/s² (models
            acceleration uncertainty).
        sigma_meas: Measurement noise standard deviation in pixels (models
            the ±1-pixel quantisation jitter of the thermal sensor).
        max_age: Maximum number of consecutive missed frames before a track
            is pruned.
        max_assoc_dist: Maximum pixel distance for greedy nearest-neighbour
            association.  Detections farther than this from any track seed
            new tracks.
    """

    def __init__(
        self,
        dt: float = 1.0 / 8.0,
        *,
        sigma_process: float = 5.0,
        sigma_meas: float = 1.5,
        max_age: int = 4,
        max_assoc_dist: float = 8.0,
    ) -> None:
        self._dt = float(dt)
        self._Q = _process_noise(sigma_process)
        self._R = np.diag([sigma_meas ** 2, sigma_meas ** 2])
        self._max_age = int(max_age)
        self._max_dist = float(max_assoc_dist)
        self._tracks: dict[int, KalmanTrack] = {}
        self._next_id: int = 0

    # ---- Public API ---------------------------------------------------------

    @property
    def tracks(self) -> dict[int, KalmanTrack]:
        return self._tracks

    def predict_all(self, dt: Optional[float] = None) -> None:
        """Advance all live tracks by dt seconds.  Call once per frame."""
        dt = float(dt) if dt is not None else self._dt
        Q = _process_noise_for_dt(dt)
        for track in self._tracks.values():
            track.predict(dt, Q)

    def update(
        self,
        detection_centroids: list[tuple[float, float]],
    ) -> dict[int, KalmanTrack]:
        """Associate measurements to tracks and update covariances.

        Uses greedy nearest-neighbour: sort detections by distance to
        nearest predicted track and assign greedily.  Unmatched detections
        create new tracks; unmatched tracks accumulate misses.

        Args:
            detection_centroids: (u, v) pixel centroids of the current
                detections for this camera (e.g. from HumanDetector).

        Returns:
            The current track dict (same reference as ``self.tracks``).
        """
        unmatched_dets = list(range(len(detection_centroids)))
        matched_track_ids: set[int] = set()

        # Keep a stable copy of the original centroids so indices stay valid
        original = list(detection_centroids)
        avail_dets: list[int] = list(range(len(original)))

        if self._tracks and original:
            track_ids = list(self._tracks.keys())
            track_pos = np.array(
                [self._tracks[tid].position for tid in track_ids], dtype=np.float64
            )
            det_pos = np.array(original, dtype=np.float64)
            # (n_dets, n_tracks) Euclidean distances
            diffs = det_pos[:, None, :] - track_pos[None, :, :]
            dist_mat = np.sqrt((diffs ** 2).sum(axis=2))

            avail_tracks = list(range(len(track_ids)))

            # Greedy assignment over the sub-matrix of still-available pairs
            while avail_dets and avail_tracks:
                sub = dist_mat[np.ix_(avail_dets, avail_tracks)]
                min_idx = int(np.argmin(sub))
                di_sub, ti_sub = divmod(min_idx, sub.shape[1])
                if sub[di_sub, ti_sub] > self._max_dist:
                    break
                di = avail_dets[di_sub]
                ti = avail_tracks[ti_sub]
                track = self._tracks[track_ids[ti]]
                z = np.array(original[di], dtype=np.float64)
                track.update(z, self._R)
                matched_track_ids.add(track_ids[ti])
                avail_dets.remove(di)
                avail_tracks.remove(ti)

        # Increment misses for unmatched tracks; prune old ones
        to_delete = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.misses += 1
                if track.misses > self._max_age:
                    to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]

        # Birth new tracks for unmatched detections
        for di in avail_dets:
            u, v = original[di]
            self._birth(u, v)

        return self._tracks

    def reset(self) -> None:
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 0

    # ---- Internal -----------------------------------------------------------

    def _birth(self, u: float, v: float) -> KalmanTrack:
        tid = self._next_id
        self._next_id += 1
        track = KalmanTrack(
            track_id=tid,
            state=np.array([u, v, 0.0, 0.0], dtype=np.float64),
            cov=np.diag([4.0, 4.0, 25.0, 25.0]),  # initial covariance
        )
        self._tracks[tid] = track
        return track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(dt: float) -> np.ndarray:
    F = np.eye(4, dtype=np.float64)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _process_noise(sigma: float) -> np.ndarray:
    """Diagonal process noise for nominal dt."""
    return np.diag([sigma ** 2, sigma ** 2, sigma ** 2, sigma ** 2])


def _process_noise_for_dt(dt: float, base_sigma: float = 5.0) -> np.ndarray:
    """Scale process noise by dt (larger step → more uncertainty)."""
    s = base_sigma * dt
    return np.diag([s ** 2, s ** 2, s ** 2, s ** 2])
