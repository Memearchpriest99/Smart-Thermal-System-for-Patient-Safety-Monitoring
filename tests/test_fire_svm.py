"""Tests for FireSVMDetector (§ 4.4.4 Feature-Based ML Approach)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import FireAlert, FireLevel, Frame
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fire_frame(profile=MLX90640, ts: float = 0.0, hot: float = 70.0) -> Frame:
    """Frame with a large, hot blob — labelled ACTIVE_COMBUSTION."""
    w, h = profile.resolution
    data = np.full((h, w), 25.0, dtype=np.float32)
    data[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = hot
    return Frame(data=data, timestamp=ts)


def _make_safe_frame(profile=MLX90640, ts: float = 0.0) -> Frame:
    """Approximately uniform frame — labelled SAFE."""
    w, h = profile.resolution
    rng = np.random.default_rng(42)
    data = (25.0 + rng.standard_normal((h, w)) * 0.3).astype(np.float32)
    return Frame(data=data, timestamp=ts)


def _make_training_set(n_fire: int = 10, n_safe: int = 10, profile=MLX90640):
    """Synthetic balanced training set."""
    rng = np.random.default_rng(0)
    frames: list[Frame] = []
    alerts: list[FireAlert] = []

    for i in range(n_fire):
        hot = rng.uniform(65.0, 80.0)
        frames.append(_make_fire_frame(profile=profile, ts=float(i), hot=hot))
        alerts.append(FireAlert(level=FireLevel.ACTIVE_COMBUSTION, timestamp=float(i)))

    for i in range(n_safe):
        w, h = profile.resolution
        noise = (25.0 + rng.standard_normal((h, w)) * 0.3).astype(np.float32)
        frames.append(Frame(data=noise, timestamp=float(n_fire + i)))
        alerts.append(FireAlert(level=FireLevel.SAFE, timestamp=float(n_fire + i)))

    return frames, alerts


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_no_profile_required(self):
        # 'invariant' algorithm — sensor_profile is optional
        det = FireSVMDetector()
        assert det.sensor_profile is None

    def test_with_profile(self):
        det = FireSVMDetector(sensor_profile=MLX90640)
        assert det.sensor_profile is MLX90640

    def test_invalid_morph_kernel_even(self):
        with pytest.raises(ValueError, match="morph_kernel_size"):
            FireSVMDetector(morph_kernel_size=4)

    def test_resolution_behavior_invariant(self):
        assert FireSVMDetector.resolution_behavior == "invariant"

    def test_is_trainable(self):
        assert FireSVMDetector.is_trainable is True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_requires_labels(self):
        det = FireSVMDetector()
        frames, _ = _make_training_set(5, 5)
        with pytest.raises(ValueError, match="y="):
            det.fit(frames, y=None)

    def test_fit_empty_dataset_raises(self):
        det = FireSVMDetector()
        with pytest.raises(ValueError):
            det.fit([], y=[])

    def test_fit_marks_fitted(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(5, 5)
        assert not det.is_fitted
        ret = det.fit(frames, alerts)
        assert det.is_fitted
        assert ret is det  # chainable

    def test_scaler_and_svm_populated_after_fit(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(5, 5)
        det.fit(frames, alerts)
        assert det._scaler is not None
        assert det._svm is not None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_before_fit_raises(self):
        det = FireSVMDetector()
        frame = _make_fire_frame()
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(frame)

    def test_output_is_fire_alert(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(10, 10)
        det.fit(frames, alerts)
        result = det.predict(_make_fire_frame())
        assert isinstance(result, FireAlert)
        assert isinstance(result.level, FireLevel)

    def test_confidence_in_unit_interval(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(10, 10)
        det.fit(frames, alerts)
        for frame in [_make_fire_frame(), _make_safe_frame()]:
            result = det.predict(frame)
            assert 0.0 <= result.confidence <= 1.0

    def test_fire_frame_classified_as_fire(self):
        # Train on 20+20 clearly separated samples; fire frames should predict fire
        det = FireSVMDetector()
        frames, alerts = _make_training_set(20, 20)
        det.fit(frames, alerts)
        fire_alert = det.predict(_make_fire_frame(hot=75.0))
        assert fire_alert.level == FireLevel.ACTIVE_COMBUSTION

    def test_timestamp_propagated(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(5, 5)
        det.fit(frames, alerts)
        result = det.predict(_make_fire_frame(ts=99.9))
        assert result.timestamp == pytest.approx(99.9)

    def test_blob_features_contain_expected_keys(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(5, 5)
        det.fit(frames, alerts)
        result = det.predict(_make_fire_frame())
        for key in ("max_temp", "mean_temp", "std_temp", "area", "skewness", "kurtosis"):
            assert key in result.blob_features

    def test_safe_frame_classified_as_safe(self):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(20, 20)
        det.fit(frames, alerts)
        # Uniform noise frame → expected SAFE
        safe_alert = det.predict(_make_safe_frame())
        assert safe_alert.level == FireLevel.SAFE

    def test_invariant_across_sensor_profiles(self):
        # A model trained without a profile should work on any sensor's frames
        det = FireSVMDetector()
        frames, alerts = _make_training_set(10, 10, profile=MLX90640)
        det.fit(frames, alerts)

        w_ws, h_ws = WAVESHARE_26984.resolution
        data = np.full((h_ws, w_ws), 25.0, dtype=np.float32)
        data[h_ws // 4: 3 * h_ws // 4, w_ws // 4: 3 * w_ws // 4] = 70.0
        ws_frame = Frame(data=data, timestamp=0.0)
        result = det.predict(ws_frame)
        assert isinstance(result.level, FireLevel)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        det = FireSVMDetector(kernel="rbf", svm_C=2.0)
        frames, alerts = _make_training_set(10, 10)
        det.fit(frames, alerts)

        path = tmp_path / "fire_svm.thalg"
        det.save(path)
        loaded = FireSVMDetector.load(path)

        assert loaded.is_fitted
        assert loaded._kernel == "rbf"
        assert loaded._svm_C == pytest.approx(2.0)

    def test_predictions_match_after_load(self, tmp_path):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(10, 10)
        det.fit(frames, alerts)

        path = tmp_path / "fire_svm.thalg"
        det.save(path)
        loaded = FireSVMDetector.load(path)

        test_frame = _make_fire_frame(ts=0.5)
        orig = det.predict(test_frame)
        restored = loaded.predict(test_frame)
        assert orig.level == restored.level
        assert orig.confidence == pytest.approx(restored.confidence, abs=1e-6)

    def test_load_wrong_class_raises(self, tmp_path):
        det = FireSVMDetector()
        frames, alerts = _make_training_set(5, 5)
        det.fit(frames, alerts)
        path = tmp_path / "fire_svm.thalg"
        det.save(path)

        with pytest.raises(TypeError):
            OtsuFireDetector = __import__(
                "thermal_algorithms.fire_detection.otsu_pipeline",
                fromlist=["OtsuFireDetector"],
            ).OtsuFireDetector
            OtsuFireDetector.load(path)

    def test_unfitted_save_load(self, tmp_path):
        det = FireSVMDetector(svm_C=3.0)
        # fit() not called — still can save (non-trainable fit semantics don't apply here,
        # but FireSVMDetector.is_trainable=True so registry would refuse; direct save is ok)
        path = tmp_path / "unfitted.thalg"
        det.save(path)
        loaded = FireSVMDetector.load(path)
        assert not loaded.is_fitted
