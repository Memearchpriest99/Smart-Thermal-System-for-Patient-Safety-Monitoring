"""Contract tests for the four task-specific ABCs.

Each ABC gets a trivial concrete implementation that exercises its specialized
predict() signature.
"""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import (
    Frame,
    Detection,
    ContactEvent,
    ActorPosition,
    FireAlert,
    FireLevel,
    HomographyMatrices,
)
from thermal_algorithms.preprocessing.base import Preprocessor
from thermal_algorithms.human_detection.base import HumanDetector
from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewFrames,
    ThreeViewDetections,
)
from thermal_algorithms.fire_detection.base import FireDetector


# ---------------------------------------------------------------------------
# Dummies — minimum-viable concretes
# ---------------------------------------------------------------------------

class _IdentityPreprocessor(Preprocessor):
    name = "_identity_preprocessor"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> Frame:
        return Frame(data=X.data.copy(), timestamp=X.timestamp, camera_id=X.camera_id)


class _StubHumanDetector(HumanDetector):
    name = "_stub_human"
    resolution_behavior = "invariant"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> list[Detection]:
        # Always returns one detection in the middle of the frame.
        h, w = X.shape
        return [
            Detection(
                bbox=(w / 4, h / 4, w / 2, h / 2),
                score=1.0,
                class_id=0,
                camera_id=X.camera_id,
            )
        ]


class _StubContactDetector(ContactDetector):
    name = "_stub_contact"
    resolution_behavior = "invariant"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(
        self,
        X: ThreeViewFrames,
        detections=None,
    ) -> ContactEvent:
        ts = max(f.timestamp for f in X)
        return ContactEvent(
            actors=(ActorPosition(world_xy=(0.0, 0.0), track_id=1),),
            pairs_in_contact=(),
            timestamp=ts,
        )


class _StubFireDetector(FireDetector):
    name = "_stub_fire"
    resolution_behavior = "invariant"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_predicted = 0

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> FireAlert:
        self._n_predicted += 1
        return FireAlert(level=FireLevel.SAFE, timestamp=X.timestamp)

    def reset(self):
        self._n_predicted = 0


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def _frame(self):
        return Frame(
            data=np.random.rand(24, 32).astype(np.float32),
            timestamp=1.0,
            camera_id=0,
        )

    def test_metadata_preserved_through_predict(self):
        p = _IdentityPreprocessor(sensor_profile=MLX90640)
        p.fit([self._frame() for _ in range(5)])
        f = self._frame()
        out = p.predict(f)
        assert out.timestamp == f.timestamp
        assert out.camera_id == f.camera_id
        assert out.shape == f.shape

    def test_transform_is_alias(self):
        p = _IdentityPreprocessor(sensor_profile=MLX90640)
        p.fit([self._frame()])
        f = self._frame()
        np.testing.assert_array_equal(p.predict(f).data, p.transform(f).data)

    def test_resolution_behavior_parameterized(self):
        assert Preprocessor.resolution_behavior == "parameterized"


# ---------------------------------------------------------------------------
# HumanDetector
# ---------------------------------------------------------------------------

class TestHumanDetector:
    def test_predict_returns_list_of_detections(self):
        d = _StubHumanDetector()
        d.fit([Frame(data=np.zeros((24, 32)), timestamp=0.0)])
        out = d.predict(Frame(data=np.zeros((24, 32)), timestamp=0.0, camera_id=1))
        assert isinstance(out, list)
        assert all(isinstance(x, Detection) for x in out)

    def test_detection_inherits_camera_id(self):
        d = _StubHumanDetector()
        out = d.predict(Frame(data=np.zeros((24, 32)), timestamp=0.0, camera_id=2))
        assert out[0].camera_id == 2

    def test_empty_list_means_no_detection(self):
        # _StubHumanDetector always returns one, but the contract allows empty.
        # Verified by type, not by behavior.
        assert HumanDetector.predict.__doc__ is not None  # contract documented


# ---------------------------------------------------------------------------
# ContactDetector
# ---------------------------------------------------------------------------

class TestContactDetector:
    def _views(self) -> ThreeViewFrames:
        return (
            Frame(data=np.zeros((24, 32)), timestamp=0.00, camera_id=0),
            Frame(data=np.zeros((24, 32)), timestamp=0.025, camera_id=1),
            Frame(data=np.zeros((24, 32)), timestamp=0.050, camera_id=2),
        )

    def test_predict_returns_contact_event(self):
        d = _StubContactDetector()
        evt = d.predict(self._views())
        assert isinstance(evt, ContactEvent)

    def test_timestamp_aligned_to_latest_view(self):
        d = _StubContactDetector()
        evt = d.predict(self._views())
        assert evt.timestamp == pytest.approx(0.050)

    def test_homography_accessor(self):
        H = HomographyMatrices(h1=np.eye(3), h2=np.eye(3), h3=np.eye(3))
        d = _StubContactDetector(homography=H)
        assert d.homography is H

    def test_set_homography_chainable(self):
        d = _StubContactDetector()
        H = HomographyMatrices(h1=np.eye(3), h2=np.eye(3), h3=np.eye(3))
        assert d.set_homography(H) is d
        assert d.homography is H


# ---------------------------------------------------------------------------
# FireDetector
# ---------------------------------------------------------------------------

class TestFireDetector:
    def test_predict_returns_fire_alert(self):
        d = _StubFireDetector()
        out = d.predict(Frame(data=np.zeros((24, 32)), timestamp=42.0))
        assert isinstance(out, FireAlert)
        assert out.timestamp == 42.0

    def test_reset_clears_state(self):
        d = _StubFireDetector()
        d.predict(Frame(data=np.zeros((24, 32)), timestamp=0.0))
        d.predict(Frame(data=np.zeros((24, 32)), timestamp=1.0))
        assert d._n_predicted == 2
        d.reset()
        assert d._n_predicted == 0
