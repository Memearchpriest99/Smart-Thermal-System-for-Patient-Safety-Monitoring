"""Tests for OtsuFireDetector (§ 4.4.4 Pipelined Approach)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import FireAlert, FireLevel, Frame
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_frame(
    value: float = 25.0,
    profile=MLX90640,
    timestamp: float = 0.0,
    camera_id: int = 0,
) -> Frame:
    """Uniform frame (all pixels same temperature — passes variance gate only if
    the caller sets a value that forces ΔT ≥ noise_floor, which it won't)."""
    w, h = profile.resolution
    return Frame(
        data=np.full((h, w), value, dtype=np.float32),
        timestamp=timestamp,
        camera_id=camera_id,
    )


def _frame_with_hot_region(
    bg: float = 25.0,
    hot_val: float = 50.0,
    hot_rows: slice = slice(11, 13),
    hot_cols: slice = slice(15, 17),
    profile=MLX90640,
    timestamp: float = 0.0,
    seed: int = 42,
) -> Frame:
    """Background frame with sensor noise + a rectangular hot spot.

    Adding noise (σ=0.5 °C) prevents the degenerate case where a perfectly
    uniform background causes Otsu to pick the minimum value as threshold,
    which would segment the entire frame as foreground.
    """
    w, h = profile.resolution
    rng = np.random.default_rng(seed)
    data = (bg + rng.standard_normal((h, w)) * 0.5).astype(np.float32)
    data[hot_rows, hot_cols] = hot_val
    return Frame(data=data, timestamp=timestamp)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_requires_sensor_profile(self):
        with pytest.raises((ValueError, TypeError)):
            OtsuFireDetector(sensor_profile=None)

    def test_invalid_morph_kernel_even(self):
        with pytest.raises(ValueError, match="morph_kernel_size"):
            OtsuFireDetector(sensor_profile=MLX90640, morph_kernel_size=4)

    def test_a_limit_auto_derived(self):
        det = OtsuFireDetector(sensor_profile=MLX90640)
        w, h = MLX90640.resolution
        expected = max(1, int(round(0.03 * w * h)))
        assert det._a_limit == expected

    def test_a_limit_explicit(self):
        det = OtsuFireDetector(sensor_profile=MLX90640, a_limit=20)
        assert det._a_limit == 20

    def test_fit_marks_fitted(self):
        det = OtsuFireDetector(sensor_profile=MLX90640)
        assert not det.is_fitted
        ret = det.fit([])
        assert det.is_fitted
        assert ret is det  # chainable


# ---------------------------------------------------------------------------
# Stage 1a: Variance gating
# ---------------------------------------------------------------------------

class TestVarianceGating:
    def test_uniform_frame_is_safe(self):
        det = OtsuFireDetector(sensor_profile=MLX90640).fit([])
        # Uniform frame → ΔT = 0 < noise_floor_c = 1.5
        alert = det.predict(_flat_frame(value=25.0))
        assert alert.level == FireLevel.SAFE
        assert alert.blob_features.get("variance_gated") == 1.0

    def test_near_uniform_frame_is_safe(self):
        det = OtsuFireDetector(sensor_profile=MLX90640).fit([])
        w, h = MLX90640.resolution
        # ΔT = 1.0 < noise_floor_c = 1.5 → still gated
        data = np.linspace(25.0, 26.0, h * w, dtype=np.float32).reshape(h, w)
        alert = det.predict(Frame(data=data, timestamp=0.0))
        assert alert.level == FireLevel.SAFE

    def test_sufficient_dynamic_range_passes_gate(self):
        det = OtsuFireDetector(sensor_profile=MLX90640, t_ign=30.0).fit([])
        # ΔT = 20 > noise_floor_c = 1.5 → gate passes
        frame = _frame_with_hot_region(bg=25.0, hot_val=45.0)
        alert = det.predict(frame)
        # Just verify it didn't return variance_gated
        assert alert.blob_features.get("variance_gated") != 1.0


# ---------------------------------------------------------------------------
# Stage 2: Decision Tree Classifier
# ---------------------------------------------------------------------------

class TestDecisionTree:
    def test_small_hot_blob_is_ignition_source(self):
        # a_limit=100: a 5×5 hot region survives erosion as a 3×3, becomes ~7×7=49px
        # after 2× dilation, which is < a_limit(100).  With background noise Otsu
        # correctly segments only the hot region.
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=45.0, t_fire=60.0, a_limit=100
        ).fit([])
        frame = _frame_with_hot_region(
            bg=25.0, hot_val=50.0,
            hot_rows=slice(10, 15), hot_cols=slice(14, 19),  # 5×5 region
        )
        alert = det.predict(frame)
        assert alert.level == FireLevel.IGNITION_SOURCE

    def test_ignition_clears_trackers(self):
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=45.0, t_fire=60.0, a_limit=10
        ).fit([])
        # Plant a "potential fire" track first
        det._tracked_blobs[99] = object()  # type: ignore[assignment]
        w, h = MLX90640.resolution
        data = np.full((h, w), 25.0, dtype=np.float32)
        data[12, 16] = 50.0
        det.predict(Frame(data=data, timestamp=0.0))
        assert len(det._tracked_blobs) == 0

    def test_safe_when_below_t_ign(self):
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=50.0, t_fire=60.0
        ).fit([])
        # Hot blob exists but T_max < t_ign → classified safe
        frame = _frame_with_hot_region(bg=25.0, hot_val=45.0)
        alert = det.predict(frame)
        assert alert.level == FireLevel.SAFE


# ---------------------------------------------------------------------------
# Stage 3: Temporal Mass Gradient (growth tracking)
# ---------------------------------------------------------------------------

class TestTemporalTracker:
    def _make_blob_frame(
        self, bg: float, hot: float, half: int, profile=MLX90640, ts: float = 0.0
    ) -> Frame:
        """Centered square hot region of radius `half` pixels with sensor noise.

        Using `half` (not `size`) directly ensures the exact blob dimensions.
        A (2*half+1)×(2*half+1) hot region is placed at the frame center.
        Background noise (σ=0.5 °C) ensures Otsu's method correctly segments
        the hot region rather than the entire frame.
        """
        w, h = profile.resolution
        rng = np.random.default_rng(42)          # fixed seed for reproducibility
        data = (bg + rng.standard_normal((h, w)) * 0.5).astype(np.float32)
        cy, cx = h // 2, w // 2
        r0 = max(0, cy - half); r1 = min(h, cy + half + 1)
        c0 = max(0, cx - half); c1 = min(w, cx + half + 1)
        data[r0:r1, c0:c1] = hot
        return Frame(data=data, timestamp=ts)

    def test_static_blob_becomes_safe_after_t_measure(self):
        # half=3 → 7×7 hot region.  After 1×erosion (3×3 kernel): 5×5 survives.
        # After 2×dilation: 9×9 = 81 px.  T_max=65>T_fire=60, area=81>a_limit=2 → PF.
        fps = MLX90640.sample_rate_hz   # 8 Hz
        det = OtsuFireDetector(
            sensor_profile=MLX90640,
            t_ign=45.0, t_fire=60.0, a_limit=2,
            delta_t=1.0,
            epsilon_growth=1.0,
            tau_step=4,
            t_measure=2.0,      # short window for test speed
            s_threshold=100,    # unreachable → blob expires via T_measure
        ).fit([])

        # Run exactly t_measure*fps + 1 frames (timeout triggers at the last frame)
        n_frames = int(fps * det._t_measure) + 1
        got_safe = False
        for i in range(n_frames):
            frame = self._make_blob_frame(bg=25.0, hot=65.0, half=3, ts=i / fps)
            alert = det.predict(frame)
            if alert.level == FireLevel.SAFE:
                got_safe = True
                break

        assert got_safe, (
            "Expected SAFE after T_measure for a static non-growing blob."
        )

    def test_growing_blob_reaches_active_combustion(self):
        # Design:
        #   k_lag = 8 frames (Δt=1s at 8Hz).
        #   Each tau_step (4 frames) the blob grows by one ring: half increases by 1.
        #   After morphological processing a (2*half+1)^2 hot region becomes
        #   approximately (2*half+3)^2 pixels  (1×erosion shrinks by 1, 2×dilation
        #   expands by 2, net +1 on each side).
        #   So area at frame i ≈ (2*(half_0 + i//tau_step) + 3)^2.
        #   At frame k_lag the gradient compares current area vs area k_lag frames ago
        #   → consistent growth → S increments toward s_threshold.
        fps = MLX90640.sample_rate_hz   # 8 Hz
        k_lag = int(round(1.0 * fps))   # 8 frames
        tau_step = 4
        s_threshold = 2
        det = OtsuFireDetector(
            sensor_profile=MLX90640,
            t_ign=45.0, t_fire=60.0, a_limit=2,
            delta_t=1.0,
            epsilon_growth=0.5,
            tau_step=tau_step,
            t_measure=30.0,
            s_threshold=s_threshold,
        ).fit([])

        # Start with half=3 (7×7) and grow by 1 ring every tau_step frames.
        # After k_lag + s_threshold*tau_step frames, S should reach s_threshold.
        n_frames = k_lag + (s_threshold + 1) * tau_step + 2
        levels = []
        for i in range(n_frames):
            half = 3 + i // tau_step   # grows: 3, 3, 3, 3, 4, 4, 4, 4, 5, ...
            # Keep blob within frame bounds (MLX90640 = 32×24, cy=12, cx=16)
            half = min(half, 5)
            frame = self._make_blob_frame(bg=25.0, hot=65.0, half=half, ts=i / fps)
            alert = det.predict(frame)
            levels.append(alert.level)

        assert FireLevel.ACTIVE_COMBUSTION in levels, (
            f"Expected ACTIVE_COMBUSTION in levels; got {set(levels)}"
        )

    def test_reset_clears_tracker_state(self):
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=45.0, t_fire=60.0, a_limit=2
        ).fit([])
        w, h = MLX90640.resolution
        data = np.full((h, w), 25.0, dtype=np.float32)
        data[10:13, 14:17] = 65.0
        det.predict(Frame(data=data, timestamp=0.0))
        assert det._frame_n == 1

        det.reset()
        assert det._frame_n == 0
        assert len(det._tracked_blobs) == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=44.0, t_fire=58.0, a_limit=15
        ).fit([])

        path = tmp_path / "otsu.thalg"
        det.save(path)

        loaded = OtsuFireDetector.load(path)
        assert loaded._t_ign == 44.0
        assert loaded._t_fire == 58.0
        assert loaded._a_limit == 15
        assert loaded.is_fitted

    def test_predict_after_load(self, tmp_path):
        det = OtsuFireDetector(
            sensor_profile=MLX90640, t_ign=45.0, t_fire=60.0, a_limit=10
        ).fit([])
        path = tmp_path / "otsu.thalg"
        det.save(path)
        loaded = OtsuFireDetector.load(path)

        w, h = MLX90640.resolution
        data = np.full((h, w), 25.0, dtype=np.float32)
        data[12, 16] = 50.0
        alert = loaded.predict(Frame(data=data, timestamp=1.0))
        assert isinstance(alert, FireAlert)

    def test_load_wrong_class_raises(self, tmp_path):
        from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector

        det = OtsuFireDetector(sensor_profile=MLX90640).fit([])
        path = tmp_path / "otsu.thalg"
        det.save(path)

        with pytest.raises(TypeError):
            FireSVMDetector.load(path)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    def test_alert_has_timestamp(self):
        det = OtsuFireDetector(sensor_profile=MLX90640).fit([])
        frame = _flat_frame(timestamp=42.5)
        alert = det.predict(frame)
        assert alert.timestamp == pytest.approx(42.5)

    def test_alert_is_frozen_dataclass(self):
        det = OtsuFireDetector(sensor_profile=MLX90640).fit([])
        alert = det.predict(_flat_frame())
        with pytest.raises((AttributeError, TypeError)):
            alert.level = FireLevel.SAFE  # type: ignore[misc]

    def test_waveshare_profile(self):
        det = OtsuFireDetector(sensor_profile=WAVESHARE_26984).fit([])
        w, h = WAVESHARE_26984.resolution
        data = np.full((h, w), 25.0, dtype=np.float32)
        frame = Frame(data=data, timestamp=0.0)
        alert = det.predict(frame)
        assert isinstance(alert.level, FireLevel)
