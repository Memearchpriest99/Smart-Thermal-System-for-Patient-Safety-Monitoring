"""Contract tests for the shared dataclasses in thermal_algorithms.core.types."""

import numpy as np
import pytest

from thermal_algorithms.core.types import (
    Frame,
    Detection,
    ActorPosition,
    ContactEvent,
    FireAlert,
    FireLevel,
    HomographyMatrices,
)


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

class TestFrame:
    def test_construct_2d(self):
        f = Frame(data=np.zeros((24, 32), dtype=np.float32), timestamp=1.0, camera_id=0)
        assert f.shape == (24, 32)
        assert f.timestamp == 1.0
        assert f.camera_id == 0

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            Frame(data=np.zeros((3, 24, 32)), timestamp=0.0)
        with pytest.raises(ValueError, match="2-D"):
            Frame(data=np.zeros(24), timestamp=0.0)

    def test_is_frozen(self):
        f = Frame(data=np.zeros((24, 32)), timestamp=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            f.timestamp = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestDetection:
    def test_foot_point_matches_report_formula(self):
        # § 4.4.3 Stage 1: P_foot = (x + w/2, y + h)
        d = Detection(bbox=(10.0, 20.0, 8.0, 6.0))
        assert d.foot_point == (14.0, 26.0)

    def test_center(self):
        d = Detection(bbox=(10.0, 20.0, 8.0, 6.0))
        assert d.center == (14.0, 23.0)

    def test_area(self):
        d = Detection(bbox=(0.0, 0.0, 8.0, 6.0))
        assert d.area == 48.0

    def test_defaults(self):
        d = Detection(bbox=(0.0, 0.0, 1.0, 1.0))
        assert d.score == 1.0
        assert d.class_id == 0
        assert d.velocity is None


# ---------------------------------------------------------------------------
# ActorPosition / ContactEvent
# ---------------------------------------------------------------------------

class TestContactEvent:
    def test_any_contact_true(self):
        actors = (
            ActorPosition(world_xy=(0.0, 0.0), track_id=1),
            ActorPosition(world_xy=(0.3, 0.0), track_id=2),
        )
        evt = ContactEvent(
            actors=actors,
            pairs_in_contact=((0, 1),),
            timestamp=10.0,
        )
        assert evt.any_contact is True

    def test_any_contact_false(self):
        actors = (
            ActorPosition(world_xy=(0.0, 0.0), track_id=1),
            ActorPosition(world_xy=(3.0, 0.0), track_id=2),
        )
        evt = ContactEvent(actors=actors, pairs_in_contact=(), timestamp=10.0)
        assert evt.any_contact is False


# ---------------------------------------------------------------------------
# FireAlert / FireLevel
# ---------------------------------------------------------------------------

class TestFireAlert:
    @pytest.mark.parametrize(
        "level, alarm",
        [
            (FireLevel.SAFE, False),
            (FireLevel.POTENTIAL_FIRE, False),
            (FireLevel.IGNITION_SOURCE, True),
            (FireLevel.ACTIVE_COMBUSTION, True),
        ],
    )
    def test_is_alarm(self, level, alarm):
        a = FireAlert(level=level, timestamp=1.0)
        assert a.is_alarm is alarm

    def test_default_features_empty(self):
        a = FireAlert(level=FireLevel.SAFE, timestamp=1.0)
        assert a.blob_features == {}


# ---------------------------------------------------------------------------
# HomographyMatrices
# ---------------------------------------------------------------------------

class TestHomographyMatrices:
    def test_construct_and_index(self):
        h1, h2, h3 = np.eye(3), np.eye(3) * 2, np.eye(3) * 3
        H = HomographyMatrices(h1=h1, h2=h2, h3=h3)
        np.testing.assert_array_equal(H[0], h1)
        np.testing.assert_array_equal(H[1], h2)
        np.testing.assert_array_equal(H[2], h3)

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="3x3"):
            HomographyMatrices(h1=np.eye(4), h2=np.eye(3), h3=np.eye(3))

    def test_as_tuple(self):
        h1, h2, h3 = np.eye(3), np.eye(3) * 2, np.eye(3) * 3
        H = HomographyMatrices(h1=h1, h2=h2, h3=h3)
        t = H.as_tuple()
        assert len(t) == 3
        np.testing.assert_array_equal(t[0], h1)
