"""OtsuFireDetector — three-stage pipelined fire detector (§ 4.4.4).

Pipeline stages
---------------
1. Adaptive Segmentation (§ 4.4.4 step 1)
   a. Variance gating: if ΔT = T_max − T_min < ΔT_nom → SAFE immediately.
   b. Otsu's method: find k* maximising between-class variance σ_B²(k).
   c. Morphological shaping: 1× erosion + 2× dilation (3×3 kernel).

2. Hierarchical Rule-Based Classifier (§ 4.4.4 step 2)
   For each blob in the ROI mask, extract area A and max temperature T_max:

       C(b) = Ignition Source   if T_max > T_ign  AND A < A_limit   → immediate Alert
            = Potential Fire    if T_max > T_fire AND A ≥ A_limit   → verify growth
            = Safe              otherwise

3. Temporal Mass Gradient Analysis (§ 4.4.4 step 3)
   Triggered only for Potential Fire blobs. Tracks blob area A_n over time.
   Frame lag: k = ⌊Δt · f_s⌋.  Discrete gradient: G_n = (A_n − A_{n−k}) / Δt.
   Persistence counter S_n:
       S_n = S_{n−1} + 1  if G_n > ε_growth   (growth frame)
           = S_{n−1} − 1  if G_n ≤ ε_growth   (static/shrinking frame, min 0)
   Decision:
       S_n ≥ S_threshold → ACTIVE_COMBUSTION  (alarm)
       elapsed > T_measure without reaching S_threshold → SAFE (static heat source)

State
-----
The detector is stateful across predict() calls. Call reset() between independent
sequences (e.g., when switching rooms or resuming after a gap).

Default thresholds
------------------
Based on §4.4.4 and the EDA in §5.2:
  t_ign   = 45 °C   (lower bound for concentrated ignition sources)
  t_fire  = 60 °C   (higher cutoff for larger heat sources)
  a_limit = auto    (≈3% of frame pixels; ~23 px for MLX, ~149 px for Waveshare)
  delta_t = 1.0 s
  k       = ⌊1.0 × 8⌋ = 8 frames  (for MLX90640 at 8 Hz)
  tau_step = 4 frames
  T_measure = 8.0 s
  S_threshold = 3

All temporal parameters remain user-tunable as the project moves to field testing.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import cv2
import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import FireAlert, FireLevel, Frame
from thermal_algorithms.fire_detection.base import FireDetector


# ---------------------------------------------------------------------------
# Hot-pixel helpers (§4.4.4 step 1, absolute-threshold path)
# ---------------------------------------------------------------------------

def _despike(data: np.ndarray, hot_c: float = 100.0) -> np.ndarray:
    """Replace isolated dead-pixel spikes with their 3×3 neighbourhood median.

    MLX90640 sensors emit occasional single-pixel glitches reading 800–935 °C.
    A pixel above ``hot_c`` whose neighbourhood median is below ``hot_c`` is an
    isolated spike (real fire is spatially coherent) and is corrected. Returns
    the input unchanged when no spike is present.
    """
    if not (data > hot_c).any():
        return data
    med = cv2.medianBlur(data.astype(np.float32), 3)
    spike = (data > hot_c) & (med < hot_c)
    if not spike.any():
        return data
    out = data.copy()
    out[spike] = med[spike]
    return out


def _extract_hot_blobs(data: np.ndarray, t_ign: float) -> list[dict]:
    """Segment by absolute temperature (``data ≥ t_ign``) and return blobs.

    Unlike :func:`otsu_utils.extract_blobs`, this uses connected-component
    *pixel counts* for area (so a single hot pixel has area 1, not 0) and does
    **no** morphological erosion — at MLX90640's 32×24 resolution a genuine fire
    is frequently a single pixel that erosion would delete. Blobs are sorted by
    area descending. Each dict matches the schema used by the rule classifier
    and the Trainer IoU evaluation (``area, max_temp, mean_temp, std_temp,
    centroid_x, centroid_y, bbox``).
    """
    from scipy.stats import skew, kurtosis as kurt

    mask = (data >= t_ign).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    blobs: list[dict] = []
    for lbl in range(1, n_labels):  # skip background label 0
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        pixels = data[labels == lbl].astype(np.float64)
        if pixels.size == 0:
            continue
        bx = float(stats[lbl, cv2.CC_STAT_LEFT])
        by = float(stats[lbl, cv2.CC_STAT_TOP])
        bw = float(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = float(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
        blobs.append({
            "area": area,
            "max_temp": float(pixels.max()),
            "mean_temp": float(pixels.mean()),
            "std_temp": float(pixels.std()),
            "centroid_x": cx,
            "centroid_y": cy,
            "skewness": float(skew(pixels)) if pixels.size > 2 and pixels.std() > 1e-6 else 0.0,
            "kurtosis": float(kurt(pixels)) if pixels.size > 2 and pixels.std() > 1e-6 else 0.0,
            "bbox": (bx, by, bw, bh),
        })
    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs


# ---------------------------------------------------------------------------
# Internal blob tracker
# ---------------------------------------------------------------------------

@dataclass
class _BlobTrack:
    """Per-blob state for the Temporal Mass Gradient tracker (§ 4.4.4 step 3)."""

    track_id: int
    centroid: tuple[float, float]
    area_history: deque              # deque[(frame_index, area)]
    S: int = 0                       # persistence counter S_n
    start_frame: int = 0             # frame index when this track was created
    last_step_frame: int = -1        # last frame at which G_n was evaluated


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class OtsuFireDetector(FireDetector):
    """Three-stage pipelined fire detector (§ 4.4.4, Pipelined Approach)."""

    name = "otsu_fire_detector"
    is_trainable = False
    resolution_behavior = "parameterized"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        # Stage 2 thresholds
        t_ign: float = 45.0,
        t_fire: float = 60.0,
        a_limit: Optional[int] = None,
        # Stage 3 temporal
        delta_t: float = 1.0,
        epsilon_growth: float = 1.0,
        tau_step: int = 4,
        t_measure: float = 8.0,
        s_threshold: int = 3,
        # Morphological
        morph_kernel_size: int = 3,
        # Blob tracking
        max_centroid_dist: float = 5.0,
        # Otsu histogram
        n_bins: int = 256,
    ) -> None:
        """
        Args:
            sensor_profile: REQUIRED. Provides noise_floor_c (ΔT_nom) and
                sample_rate_hz (used to convert Δt → frame lag k).
            t_ign: T_ign in °C — lower threshold for a small concentrated
                heat source (ignition source). Default 45 °C per report.
            t_fire: T_fire in °C — upper threshold for a large heat source
                (Potential Fire). Default 60 °C per report.
            a_limit: A_limit in pixels — spatial cutoff between Ignition Source
                and Potential Fire. If None, auto-derived as ≈3% of frame area.
            delta_t: Δt in seconds — look-back window for gradient calculation.
            epsilon_growth: ε_growth — minimum pixel difference to count as
                growth. Default 1 pixel (conservative for low-res sensors).
            tau_step: τ_step in frames — interval between gradient evaluations.
                Default 4 frames (0.5 s at 8 Hz).
            t_measure: T_measure in seconds — maximum measurement period before
                a Potential Fire is reclassified as a static heat source.
            s_threshold: S_threshold — minimum accumulated growth score for
                ACTIVE_COMBUSTION. Default 3.
            morph_kernel_size: Size of the square structuring element for
                morphological operations. Must be odd ≥ 1.
            max_centroid_dist: Maximum pixel distance between consecutive frame
                centroids to consider the same blob as the same track.
            n_bins: Number of histogram bins for Otsu's method. Default 256.
        """
        if sensor_profile is None:
            raise ValueError("OtsuFireDetector requires a SensorProfile.")
        if morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
            raise ValueError(
                f"morph_kernel_size must be a positive odd integer; got {morph_kernel_size}."
            )

        if a_limit is None:
            w, h = sensor_profile.resolution
            a_limit = max(1, int(round(0.03 * w * h)))

        super().__init__(
            sensor_profile=sensor_profile,
            t_ign=t_ign,
            t_fire=t_fire,
            a_limit=a_limit,
            delta_t=delta_t,
            epsilon_growth=epsilon_growth,
            tau_step=tau_step,
            t_measure=t_measure,
            s_threshold=s_threshold,
            morph_kernel_size=morph_kernel_size,
            max_centroid_dist=max_centroid_dist,
            n_bins=n_bins,
        )

        self._t_ign = float(t_ign)
        self._t_fire = float(t_fire)
        self._a_limit = int(a_limit)
        self._delta_t = float(delta_t)
        self._epsilon = float(epsilon_growth)
        self._tau_step = int(tau_step)
        self._t_measure = float(t_measure)
        self._s_threshold = int(s_threshold)
        self._max_centroid_dist = float(max_centroid_dist)
        self._n_bins = int(n_bins)
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self._fps: float = sensor_profile.sample_rate_hz

        # Runtime state — cleared by reset()
        self._frame_n: int = 0
        self._tracked_blobs: dict[int, _BlobTrack] = {}
        self._next_track_id: int = 0

    # ---- Calibration / training ------------------------------------------

    def fit(
        self,
        X: Iterable[Frame],
        y: Iterable[FireAlert] | None = None,
    ) -> "OtsuFireDetector":
        """No-op calibration: thresholds are set at construction time.

        A future calibration pass could auto-tune t_ign / t_fire / a_limit
        from labeled data; for now fit() simply marks the detector as ready.
        """
        self._is_fitted = True
        return self

    # ---- Core inference --------------------------------------------------

    def predict(self, X: Frame) -> FireAlert:
        """Classify a single frame and update the temporal tracker.

        For predictable results in sequential processing, call predict() on
        consecutive frames in order. Call reset() when starting a new sequence.
        """
        data = _despike(X.data.astype(np.float32))
        n = self._frame_n
        self._frame_n += 1

        # ---- Stage 1a: Variance gating ------------------------------------
        delta_t_frame = float(data.max()) - float(data.min())
        noise_floor = self._sensor_profile.noise_floor_c  # type: ignore[union-attr]

        if delta_t_frame < noise_floor:
            self._tracked_blobs.clear()
            return FireAlert(
                level=FireLevel.SAFE,
                timestamp=X.timestamp,
                blob_features={"delta_t_frame": delta_t_frame, "variance_gated": 1.0},
                confidence=1.0,
            )

        # ---- Stage 1b-c: Absolute hot-pixel segmentation ------------------
        # Segment directly on `data ≥ t_ign` (connected components, no erosion)
        # rather than Otsu + morphology. At 32×24 a real fire is often a single
        # hot pixel: Otsu's global threshold is anchored by warm bodies and the
        # 1× erosion deletes the fire pixel entirely. The hierarchical
        # classifier only ever acts on blobs with max_temp ≥ t_ign anyway, so
        # thresholding at t_ign is both equivalent in intent and recovers the
        # small/single-pixel fires the morphological path discarded.
        hot_blobs = _extract_hot_blobs(data, self._t_ign)

        ignition_blobs = []
        potential_fire_blobs = []
        for b in hot_blobs:
            if b["max_temp"] > self._t_ign and b["area"] < self._a_limit:
                ignition_blobs.append(b)
            elif b["max_temp"] > self._t_fire and b["area"] >= self._a_limit:
                potential_fire_blobs.append(b)

        # Ignition source → immediate alert (no temporal confirmation required)
        if ignition_blobs:
            best = max(ignition_blobs, key=lambda b: b["max_temp"])
            self._tracked_blobs.clear()
            return FireAlert(
                level=FireLevel.IGNITION_SOURCE,
                timestamp=X.timestamp,
                blob_features=best,
                confidence=1.0,
            )

        # ---- Stage 3: Temporal Mass Gradient (only for Potential Fire) -----
        if potential_fire_blobs:
            level = self._update_trackers(potential_fire_blobs, n)
            best = max(potential_fire_blobs, key=lambda b: b["area"])
            return FireAlert(
                level=level,
                timestamp=X.timestamp,
                blob_features=best,
                confidence=1.0,
            )

        # No threatening blobs this frame → clear all trackers
        self._tracked_blobs.clear()
        return FireAlert(
            level=FireLevel.SAFE,
            timestamp=X.timestamp,
            blob_features={"n_blobs_total": float(len(hot_blobs)), "delta_t_frame": delta_t_frame},
            confidence=1.0,
        )

    def reset(self) -> None:
        """Clear the temporal tracker and frame counter."""
        self._frame_n = 0
        self._tracked_blobs.clear()
        self._next_track_id = 0

    # ---- Stage 3 internals -----------------------------------------------

    def _update_trackers(
        self, potential_blobs: list[dict], frame_n: int
    ) -> FireLevel:
        """Match blobs to tracks, update S_n counters, return highest level."""
        max_history = int(math.ceil(self._fps * self._t_measure)) + 2
        k_lag = int(round(self._delta_t * self._fps))  # frame lag

        matched_ids: set[int] = set()
        to_delete: set[int] = set()
        result_level: FireLevel = FireLevel.POTENTIAL_FIRE
        # Tracks whether any blob is still actively accumulating evidence (not
        # yet timed out). Used to downgrade POTENTIAL_FIRE → SAFE when all
        # Potential Fire candidates are resolved as static heat sources.
        any_still_tracking: bool = False

        for blob in potential_blobs:
            tid = self._nearest_track(blob)
            if tid is None:
                tid = self._next_track_id
                self._next_track_id += 1
                self._tracked_blobs[tid] = _BlobTrack(
                    track_id=tid,
                    centroid=(blob["centroid_x"], blob["centroid_y"]),
                    area_history=deque(maxlen=max_history),
                    S=0,
                    start_frame=frame_n,
                    last_step_frame=frame_n - 1,
                )

            track = self._tracked_blobs[tid]

            # Smooth centroid (exponential moving average)
            ox, oy = track.centroid
            track.centroid = (
                0.7 * ox + 0.3 * blob["centroid_x"],
                0.7 * oy + 0.3 * blob["centroid_y"],
            )
            track.area_history.append((frame_n, blob["area"]))
            matched_ids.add(tid)

            # Gradient evaluation every tau_step frames, once k_lag frames elapsed
            frames_elapsed = frame_n - track.start_frame
            frames_since_step = frame_n - track.last_step_frame
            if frames_elapsed >= k_lag and frames_since_step >= self._tau_step:
                track.last_step_frame = frame_n
                a_now = blob["area"]
                a_old = self._lookup_area(track, frame_n - k_lag)
                if a_old is not None:
                    g_n = (a_now - a_old) / self._delta_t
                    if g_n > self._epsilon:
                        track.S += 1
                    else:
                        track.S = max(0, track.S - 1)

            # Decision
            elapsed_secs = frames_elapsed / self._fps
            if track.S >= self._s_threshold:
                result_level = FireLevel.ACTIVE_COMBUSTION
            elif elapsed_secs >= self._t_measure:
                # T_measure elapsed without reaching S_threshold → static heat source
                to_delete.add(tid)
            else:
                any_still_tracking = True

        # Remove stale (unmatched) and timed-out tracks
        stale = {tid for tid in self._tracked_blobs if tid not in matched_ids}
        for tid in to_delete | stale:
            self._tracked_blobs.pop(tid, None)

        # If all Potential Fire candidates resolved without growth, downgrade to SAFE
        if result_level == FireLevel.POTENTIAL_FIRE and not any_still_tracking:
            result_level = FireLevel.SAFE

        return result_level

    def _nearest_track(self, blob: dict) -> Optional[int]:
        """Return track_id of nearest existing track within max_centroid_dist."""
        cx, cy = blob["centroid_x"], blob["centroid_y"]
        best_id: Optional[int] = None
        best_dist = self._max_centroid_dist
        for tid, track in self._tracked_blobs.items():
            tx, ty = track.centroid
            dist = math.hypot(cx - tx, cy - ty)
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def _lookup_area(self, track: _BlobTrack, target_frame: int) -> Optional[float]:
        """Return the area recorded closest to target_frame in history."""
        tolerance = max(2, self._tau_step)
        best_area: Optional[float] = None
        best_diff = tolerance + 1
        for fn, area in track.area_history:
            diff = abs(fn - target_frame)
            if diff < best_diff:
                best_diff = diff
                best_area = area
        return best_area

    # ---- Persistence -------------------------------------------------------

    def _state_dict(self) -> dict:
        return {"resolved_a_limit": self._a_limit}

    def _load_state_dict(self, state: dict) -> None:
        if "resolved_a_limit" in state:
            self._a_limit = int(state["resolved_a_limit"])
        self._is_fitted = True
