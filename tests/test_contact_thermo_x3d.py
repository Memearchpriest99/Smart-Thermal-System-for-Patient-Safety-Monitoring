"""Tests for ThermoX3DDetector (§ 4.4.3.3). Gated by torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import ContactEvent, Frame
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector, _build_x3d_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(profile=MLX90640, ts=0.0, cam=0, hot=False) -> Frame:
    w, h = profile.resolution
    data = np.random.default_rng(42).standard_normal((h, w)).astype(np.float32) * 0.5 + 25.0
    if hot:
        data[h // 4:3 * h // 4, w // 4:3 * w // 4] = 38.0  # two merged bodies
    return Frame(data=data, timestamp=ts, camera_id=cam)


def _make_triplet(profile=MLX90640, ts=0.0, hot=False) -> tuple:
    return tuple(_make_frame(profile, ts, cam=i, hot=hot) for i in range(3))


def _make_training_data(n=40, profile=MLX90640, T=4):
    frames, events = [], []
    for i in range(n):
        hot = i >= n // 2
        frames.append(_make_triplet(profile, ts=float(i), hot=hot))
        pairs = ((0, 1),) if hot else ()
        events.append(ContactEvent(actors=(), pairs_in_contact=pairs, timestamp=float(i)))
    return frames, events


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TestX3DModel:
    def test_mlx_forward_output_shape(self):
        h, w = MLX90640.height, MLX90640.width
        model = _build_x3d_model(h, w)
        x = torch.zeros(2, 3, 4, h, w)    # (B=2, cams=3, T=4, H, W)
        out = model(x)
        assert out.shape == (2, 2)

    def test_waveshare_forward_output_shape(self):
        h, w = WAVESHARE_26984.height, WAVESHARE_26984.width
        model = _build_x3d_model(h, w)
        x = torch.zeros(1, 3, 4, h, w)
        out = model(x)
        assert out.shape == (1, 2)

    def test_output_finite(self):
        h, w = MLX90640.height, MLX90640.width
        model = _build_x3d_model(h, w)
        x = torch.randn(1, 3, 4, h, w)
        out = model(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

class TestX3DConstruction:
    def test_requires_sensor_profile(self):
        with pytest.raises((ValueError, TypeError)):
            ThermoX3DDetector(sensor_profile=None)

    def test_resolution_behavior_fixed(self):
        assert ThermoX3DDetector.resolution_behavior == "fixed"

    def test_is_trainable(self):
        assert ThermoX3DDetector.is_trainable is True

    def test_not_fitted_initially(self):
        det = ThermoX3DDetector(sensor_profile=MLX90640)
        assert not det.is_fitted


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestX3DFit:
    def test_fit_requires_labels(self):
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4)
        frames, _ = _make_training_data(n=8, T=4)
        with pytest.raises(ValueError, match="y="):
            det.fit(frames, y=None)

    def test_fit_marks_fitted(self):
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4, n_epochs=1)
        frames, events = _make_training_data(n=8, T=4)
        ret = det.fit(frames, events)
        assert det.is_fitted
        assert ret is det

    def test_fit_sets_normalisation_stats(self):
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4, n_epochs=1)
        frames, events = _make_training_data(n=8, T=4)
        det.fit(frames, events)
        assert det._global_mean != 0.0 or det._global_std != 1.0


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestX3DPredict:
    def _trained_det(self, T=4) -> ThermoX3DDetector:
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=T, n_epochs=1)
        frames, events = _make_training_data(n=T + 4, T=T)
        det.fit(frames, events)
        det.reset()
        return det

    def test_buffer_filling_returns_zero_confidence(self):
        det = self._trained_det(T=8)
        event = det.predict(_make_triplet())
        assert event.confidence == pytest.approx(0.0)
        assert event.debug.get("status") == "buffer_filling"

    def test_confidence_in_unit_interval_after_full(self):
        det = self._trained_det(T=4)
        event = None
        for i in range(6):
            event = det.predict(_make_triplet(ts=float(i)))
        assert 0.0 <= event.confidence <= 1.0

    def test_ignores_detections_argument(self):
        det = self._trained_det(T=4)
        for i in range(4):
            e1 = det.predict(_make_triplet(ts=float(i)), detections=None)
        det.reset()
        for i in range(4):
            e2 = det.predict(_make_triplet(ts=float(i)), detections=([], [], []))
        # Both should return the same confidence (detections are ignored)
        assert e1.confidence == pytest.approx(e2.confidence, abs=1e-5)

    def test_reset_clears_buffers(self):
        det = self._trained_det(T=4)
        for _ in range(4):
            det.predict(_make_triplet())
        det.reset()
        event = det.predict(_make_triplet())
        assert event.confidence == pytest.approx(0.0)

    def test_timestamp_propagated(self):
        det = self._trained_det(T=4)
        event = det.predict(_make_triplet(ts=99.5))
        assert event.timestamp == pytest.approx(99.5)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestX3DPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4, n_epochs=1)
        frames, events = _make_training_data(n=8, T=4)
        det.fit(frames, events)

        path = tmp_path / "x3d.thalg"
        det.save(path)
        loaded = ThermoX3DDetector.load(path)
        assert loaded.is_fitted
        assert loaded._global_mean == pytest.approx(det._global_mean)

    def test_predictions_match_after_load(self, tmp_path):
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4, n_epochs=1)
        frames, events = _make_training_data(n=8, T=4)
        det.fit(frames, events)

        path = tmp_path / "x3d.thalg"
        det.save(path)
        loaded = ThermoX3DDetector.load(path)

        det.reset(); loaded.reset()
        event1 = event2 = None
        for i in range(6):
            event1 = det.predict(_make_triplet(ts=float(i)))
            event2 = loaded.predict(_make_triplet(ts=float(i)))
        assert event1.confidence == pytest.approx(event2.confidence, abs=1e-5)

    def test_wrong_class_raises(self, tmp_path):
        from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
        det = ThermoX3DDetector(sensor_profile=MLX90640, T=4, n_epochs=1)
        frames, events = _make_training_data(n=8, T=4)
        det.fit(frames, events)
        path = tmp_path / "x3d.thalg"
        det.save(path)
        with pytest.raises(TypeError):
            GeometricContactDetector.load(path)
