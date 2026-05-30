"""Tests for GeometricContactDetector (§ 4.4.3.1)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import (
    ActorPosition,
    ContactEvent,
    Detection,
    Frame,
    HomographyMatrices,
)
from thermal_algorithms.contact_detection.geometric import GeometricContactDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_H() -> HomographyMatrices:
    return HomographyMatrices(h1=np.eye(3), h2=np.eye(3), h3=np.eye(3))


def _make_frames(ts: float = 0.0) -> tuple:
    w, h = MLX90640.resolution
    data = np.zeros((h, w), dtype=np.float32)
    return (
        Frame(data=data, timestamp=ts, camera_id=0),
        Frame(data=data, timestamp=ts, camera_id=1),
        Frame(data=data, timestamp=ts, camera_id=2),
    )


def _det_at(world_xy: tuple[float, float], cam_id: int) -> Detection:
    """Detection whose foot_point maps to world_xy under the identity homography."""
    u, v = world_xy
    return Detection(bbox=(u - 1.0, v - 2.0, 2.0, 2.0), score=1.0, camera_id=cam_id)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs_without_homography(self):
        det = GeometricContactDetector()
        assert det.homography is None

    def test_constructs_with_homography(self):
        H = _identity_H()
        det = GeometricContactDetector(homography=H)
        assert det.homography is H

    def test_fit_marks_fitted(self):
        det = GeometricContactDetector()
        assert not det.is_fitted
        ret = det.fit([])
        assert det.is_fitted
        assert ret is det

    def test_resolution_behavior_invariant(self):
        assert GeometricContactDetector.resolution_behavior == "invariant"

    def test_not_trainable(self):
        assert GeometricContactDetector.is_trainable is False


# ---------------------------------------------------------------------------
# Predict — error cases
# ---------------------------------------------------------------------------

class TestPredictErrors:
    def test_raises_without_homography(self):
        det = GeometricContactDetector().fit([])
        frames = _make_frames()
        dets = ([_det_at((1.0, 1.0), 0)], [], [])
        with pytest.raises(RuntimeError, match="homography"):
            det.predict(frames, detections=dets)

    def test_raises_without_detections(self):
        det = GeometricContactDetector(homography=_identity_H()).fit([])
        with pytest.raises(ValueError, match="detections"):
            det.predict(_make_frames(), detections=None)


# ---------------------------------------------------------------------------
# Predict — contact logic
# ---------------------------------------------------------------------------

class TestContactLogic:
    def _detector(self, delta_m: float = 0.5) -> GeometricContactDetector:
        return GeometricContactDetector(
            homography=_identity_H(),
            epsilon_m=0.5,
            delta_m=delta_m,
        ).fit([])

    def test_no_detections_returns_no_contact(self):
        det = self._detector()
        event = det.predict(_make_frames(), detections=([], [], []))
        assert not event.any_contact
        assert event.actors == ()

    def test_single_source_detection_discarded(self):
        # Only camera 0 sees a person → no cross-validation → no actor
        det = self._detector()
        dets = ([_det_at((1.0, 1.0), 0)], [], [])
        event = det.predict(_make_frames(), detections=dets)
        assert event.actors == ()
        assert not event.any_contact

    def test_two_actors_far_apart_no_contact(self):
        det = self._detector(delta_m=0.5)
        # Two people 2 m apart, each seen by cams 0 and 1
        dets = (
            [_det_at((0.0, 0.0), 0), _det_at((2.0, 0.0), 0)],
            [_det_at((0.0, 0.0), 1), _det_at((2.0, 0.0), 1)],
            [],
        )
        event = det.predict(_make_frames(), detections=dets)
        assert len(event.actors) == 2
        assert not event.any_contact

    def test_two_actors_close_together_contact(self):
        # epsilon_m=0.15 keeps the two people as separate clusters (0.4 m > 0.15 m),
        # while delta_m=0.5 flags them as in contact (0.4 m < 0.5 m).
        det = GeometricContactDetector(
            homography=_identity_H(), epsilon_m=0.15, delta_m=0.5
        ).fit([])
        dets = (
            [_det_at((0.0, 0.0), 0), _det_at((0.4, 0.0), 0)],
            [_det_at((0.0, 0.0), 1), _det_at((0.4, 0.0), 1)],
            [],
        )
        event = det.predict(_make_frames(), detections=dets)
        assert event.any_contact
        assert (0, 1) in event.pairs_in_contact

    def test_timestamp_from_latest_frame(self):
        det = self._detector()
        w, h = MLX90640.resolution
        data = np.zeros((h, w), dtype=np.float32)
        frames = (
            Frame(data=data, timestamp=1.0, camera_id=0),
            Frame(data=data, timestamp=1.05, camera_id=1),
            Frame(data=data, timestamp=1.10, camera_id=2),
        )
        event = det.predict(frames, detections=([], [], []))
        assert event.timestamp == pytest.approx(1.10)

    def test_contact_event_is_frozen(self):
        det = self._detector()
        event = det.predict(_make_frames(), detections=([], [], []))
        with pytest.raises((AttributeError, TypeError)):
            event.any_contact = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Homography calibration
# ---------------------------------------------------------------------------

class TestHomographyCalibration:
    def test_calibrate_stores_matrices(self):
        det = GeometricContactDetector().fit([])
        H_true = np.array([
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 0.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(8):
            u, v = rng.uniform(0, 32), rng.uniform(0, 24)
            p = H_true @ np.array([u, v, 1.0])
            xw, yw = p[0] / p[2], p[1] / p[2]
            pairs.append(((float(u), float(v)), (float(xw), float(yw))))

        corr = [(k, pairs) for k in range(3)]
        H_result = det.calibrate_homography(corr)
        assert H_result is det.homography
        assert det.homography.h1.shape == (3, 3)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip_with_homography(self, tmp_path):
        det = GeometricContactDetector(
            homography=_identity_H(), epsilon_m=0.4, delta_m=0.6,
        ).fit([])
        path = tmp_path / "geometric.thalg"
        det.save(path)
        loaded = GeometricContactDetector.load(path)
        assert loaded.is_fitted
        assert loaded._epsilon_m == pytest.approx(0.4)
        assert loaded._delta_m == pytest.approx(0.6)
        assert loaded.homography is not None

    def test_save_load_predictions_match(self, tmp_path):
        det = GeometricContactDetector(
            homography=_identity_H(), delta_m=0.5
        ).fit([])
        path = tmp_path / "geo.thalg"
        det.save(path)
        loaded = GeometricContactDetector.load(path)

        dets = (
            [_det_at((0.2, 0.0), 0)],
            [_det_at((0.2, 0.0), 1)],
            [_det_at((0.2, 0.0), 2)],
        )
        e1 = det.predict(_make_frames(), detections=dets)
        e2 = loaded.predict(_make_frames(), detections=dets)
        assert e1.actors[0].world_xy == pytest.approx(e2.actors[0].world_xy, abs=0.01)
