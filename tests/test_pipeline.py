"""Tests for ThermalPipeline — the runtime integration layer."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import (
    ContactEvent,
    Detection,
    FireAlert,
    FireLevel,
    Frame,
)
from thermal_algorithms.pipeline import Alert, AlertType, PipelineResult, ThermalPipeline


# ---------------------------------------------------------------------------
# Helpers & stub components
# ---------------------------------------------------------------------------

def _frame(ts: float = 0.0, cam: int = 0) -> Frame:
    w, h = MLX90640.resolution
    data = np.full((h, w), 25.0, dtype=np.float32)
    return Frame(data=data, timestamp=ts, camera_id=cam)


def _triplet(ts: float = 0.0) -> tuple[Frame, Frame, Frame]:
    return _frame(ts, 0), _frame(ts + 0.01, 1), _frame(ts + 0.05, 2)


class _NoopPreprocessor:
    """Returns frame unchanged."""
    def predict(self, frame: Frame) -> Frame:
        return frame
    def reset(self): pass


class _NoDetections:
    def predict(self, frame: Frame) -> list:
        return []
    def reset(self): pass


class _OneDetection:
    def predict(self, frame: Frame) -> list[Detection]:
        return [Detection(bbox=(5.0, 5.0, 5.0, 10.0), score=1.0, camera_id=frame.camera_id)]
    def reset(self): pass


class _SafeFireDetector:
    def predict(self, frame: Frame) -> FireAlert:
        return FireAlert(level=FireLevel.SAFE, timestamp=frame.timestamp)
    def reset(self): pass


class _AlarmFireDetector:
    def __init__(self, level=FireLevel.IGNITION_SOURCE):
        self._level = level
    def predict(self, frame: Frame) -> FireAlert:
        return FireAlert(level=self._level, timestamp=frame.timestamp, confidence=0.9)
    def reset(self): pass


class _NoContactDetector:
    def predict(self, frames, detections=None) -> ContactEvent:
        ts = max(f.timestamp for f in frames)
        return ContactEvent(actors=(), pairs_in_contact=(), timestamp=ts)
    def reset(self): pass


class _AlwaysContactDetector:
    def predict(self, frames, detections=None) -> ContactEvent:
        from thermal_algorithms.core.types import ActorPosition
        ts = max(f.timestamp for f in frames)
        actors = (
            ActorPosition(world_xy=(0.0, 0.0), track_id=0, source_camera_ids=(0,)),
            ActorPosition(world_xy=(0.3, 0.0), track_id=1, source_camera_ids=(1,)),
        )
        return ContactEvent(actors=actors, pairs_in_contact=((0, 1),),
                            timestamp=ts, confidence=0.85)
    def reset(self): pass


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_all_optional_components(self):
        p = ThermalPipeline()
        assert p.preprocessor is None
        assert p.fire_detector is None
        assert p.contact_detector is None

    def test_restricted_default_false(self):
        assert ThermalPipeline().restricted is False

    def test_restricted_flag_set(self):
        assert ThermalPipeline(restricted=True).restricted is True

    def test_repr_includes_component_names(self):
        p = ThermalPipeline(
            preprocessor=_NoopPreprocessor(),
            fire_detector=_SafeFireDetector(),
        )
        r = repr(p)
        assert "_NoopPreprocessor" in r
        assert "_SafeFireDetector" in r

    def test_cooldown_defaults(self):
        p = ThermalPipeline()
        assert p.cooldown_s[AlertType.FIRE] == pytest.approx(30.0)
        assert p.cooldown_s[AlertType.CONTACT] == pytest.approx(30.0)

    def test_custom_cooldowns(self):
        p = ThermalPipeline(fire_cooldown_s=10.0, contact_cooldown_s=5.0)
        assert p.cooldown_s[AlertType.FIRE] == pytest.approx(10.0)
        assert p.cooldown_s[AlertType.CONTACT] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# PipelineResult structure
# ---------------------------------------------------------------------------

class TestPipelineResultStructure:
    def _run(self, **kwargs) -> PipelineResult:
        return ThermalPipeline(**kwargs).process(*_triplet(ts=0.0))

    def test_result_has_three_processed_frames(self):
        r = self._run()
        assert len(r.processed_frames) == 3

    def test_result_has_three_detection_lists(self):
        r = self._run()
        assert len(r.detections) == 3

    def test_result_has_three_fire_alerts(self):
        r = self._run()
        assert len(r.fire_alerts) == 3

    def test_timestamp_is_max_of_input_frames(self):
        p = ThermalPipeline()
        f0 = _frame(1.0, 0); f1 = _frame(1.01, 1); f2 = _frame(1.05, 2)
        r = p.process(f0, f1, f2)
        assert r.timestamp == pytest.approx(1.05)

    def test_no_fire_detector_gives_safe_alerts(self):
        p = ThermalPipeline()
        r = p.process(*_triplet())
        assert all(fa.level == FireLevel.SAFE for fa in r.fire_alerts)

    def test_no_contact_detector_gives_none_event(self):
        p = ThermalPipeline()
        r = p.process(*_triplet())
        assert r.contact_event is None

    def test_preprocessor_is_applied(self):
        class _TaggingPreprocessor:
            def predict(self, frame: Frame) -> Frame:
                return Frame(data=frame.data, timestamp=frame.timestamp,
                             camera_id=frame.camera_id,
                             metadata={**frame.metadata, "tagged": True})
            def reset(self): pass

        p = ThermalPipeline(preprocessor=_TaggingPreprocessor())
        r = p.process(*_triplet())
        assert all(f.metadata.get("tagged") for f in r.processed_frames)

    def test_result_is_frozen(self):
        r = ThermalPipeline().process(*_triplet())
        with pytest.raises((AttributeError, TypeError)):
            r.timestamp = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# any_alarm / fire_alarm / contact_alarm properties
# ---------------------------------------------------------------------------

class TestResultProperties:
    def test_any_alarm_false_on_safe_result(self):
        r = ThermalPipeline().process(*_triplet())
        assert r.any_alarm is False

    def test_fire_alarm_false_without_fire_detector(self):
        r = ThermalPipeline().process(*_triplet())
        assert r.fire_alarm is False

    def test_contact_alarm_false_without_contact_detector(self):
        r = ThermalPipeline().process(*_triplet())
        assert r.contact_alarm is False


# ---------------------------------------------------------------------------
# Fire alert path
# ---------------------------------------------------------------------------

class TestFirePath:
    def test_safe_fire_detector_no_alert(self):
        p = ThermalPipeline(fire_detector=_SafeFireDetector())
        r = p.process(*_triplet())
        assert not r.fire_alarm

    def test_alarm_fire_detector_raises_alert(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector())
        r = p.process(*_triplet())
        assert r.fire_alarm
        assert r.any_alarm

    def test_fire_alert_type(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector())
        r = p.process(*_triplet())
        assert r.alerts[0].type == AlertType.FIRE

    def test_fire_alert_confidence(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector())
        r = p.process(*_triplet())
        assert r.alerts[0].confidence == pytest.approx(0.9)

    def test_fire_alert_debug_contains_level(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector())
        r = p.process(*_triplet())
        assert "level" in r.alerts[0].debug

    def test_fire_cooldown_suppresses_second_alert(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector(), fire_cooldown_s=10.0)
        r1 = p.process(*_triplet(ts=0.0))
        r2 = p.process(*_triplet(ts=5.0))   # only 5 s later — within cooldown
        assert r1.fire_alarm
        assert not r2.fire_alarm

    def test_fire_cooldown_allows_alert_after_expiry(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector(), fire_cooldown_s=10.0)
        p.process(*_triplet(ts=0.0))
        r = p.process(*_triplet(ts=11.0))  # 11 s later — cooldown expired
        assert r.fire_alarm

    def test_fire_alert_on_first_call_always_fires(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector(), fire_cooldown_s=30.0)
        r = p.process(*_triplet(ts=0.0))
        assert r.fire_alarm

    def test_active_combustion_also_triggers_alert(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector(FireLevel.ACTIVE_COMBUSTION))
        r = p.process(*_triplet())
        assert r.fire_alarm

    def test_potential_fire_does_not_alert(self):
        # POTENTIAL_FIRE is not is_alarm — only IGNITION_SOURCE and ACTIVE_COMBUSTION
        class _PotentialFireDetector:
            def predict(self, f): return FireAlert(level=FireLevel.POTENTIAL_FIRE, timestamp=f.timestamp)
            def reset(self): pass
        p = ThermalPipeline(fire_detector=_PotentialFireDetector())
        r = p.process(*_triplet())
        assert not r.fire_alarm


# ---------------------------------------------------------------------------
# Contact alert path
# ---------------------------------------------------------------------------

class TestContactPath:
    def test_no_contact_no_alert(self):
        p = ThermalPipeline(contact_detector=_NoContactDetector())
        r = p.process(*_triplet())
        assert not r.contact_alarm

    def test_contact_raises_alert(self):
        p = ThermalPipeline(contact_detector=_AlwaysContactDetector())
        r = p.process(*_triplet())
        assert r.contact_alarm
        assert r.any_alarm

    def test_contact_alert_type(self):
        p = ThermalPipeline(contact_detector=_AlwaysContactDetector())
        r = p.process(*_triplet())
        assert r.alerts[0].type == AlertType.CONTACT

    def test_contact_alert_confidence(self):
        p = ThermalPipeline(contact_detector=_AlwaysContactDetector())
        r = p.process(*_triplet())
        assert r.alerts[0].confidence == pytest.approx(0.85)

    def test_contact_cooldown_suppresses_second_alert(self):
        p = ThermalPipeline(contact_detector=_AlwaysContactDetector(), contact_cooldown_s=20.0)
        r1 = p.process(*_triplet(ts=0.0))
        r2 = p.process(*_triplet(ts=10.0))
        assert r1.contact_alarm
        assert not r2.contact_alarm

    def test_contact_cooldown_independent_of_fire_cooldown(self):
        # A FIRE alert should not affect the CONTACT cooldown and vice versa
        p = ThermalPipeline(
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
            fire_cooldown_s=60.0,
            contact_cooldown_s=60.0,
        )
        r1 = p.process(*_triplet(ts=0.0))
        r2 = p.process(*_triplet(ts=5.0))
        assert r1.fire_alarm and r1.contact_alarm
        assert not r2.fire_alarm and not r2.contact_alarm

    def test_contact_event_stored_in_result(self):
        p = ThermalPipeline(contact_detector=_AlwaysContactDetector())
        r = p.process(*_triplet())
        assert r.contact_event is not None
        assert r.contact_event.any_contact


# ---------------------------------------------------------------------------
# Both paths active simultaneously
# ---------------------------------------------------------------------------

class TestRestrictedAreaPath:
    def test_no_human_no_alert(self):
        p = ThermalPipeline(restricted=True, human_detector=_NoDetections())
        r = p.process(*_triplet())
        assert not r.restricted_area_alarm
        assert not r.any_alarm

    def test_human_detected_triggers_alert(self):
        p = ThermalPipeline(restricted=True, human_detector=_OneDetection())
        r = p.process(*_triplet())
        assert r.restricted_area_alarm
        assert r.alerts[0].type == AlertType.RESTRICTED_AREA

    def test_fire_and_contact_detectors_not_run_in_restricted_mode(self):
        # Even if fire and contact detectors are provided, they should not run
        p = ThermalPipeline(
            restricted=True,
            human_detector=_OneDetection(),
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
        )
        r = p.process(*_triplet())
        # Only RESTRICTED_AREA, no FIRE or CONTACT
        alert_types = {a.type for a in r.alerts}
        assert AlertType.RESTRICTED_AREA in alert_types
        assert AlertType.FIRE not in alert_types
        assert AlertType.CONTACT not in alert_types

    def test_fire_alerts_are_safe_in_restricted_mode(self):
        p = ThermalPipeline(restricted=True, fire_detector=_AlarmFireDetector())
        r = p.process(*_triplet())
        assert all(fa.level == FireLevel.SAFE for fa in r.fire_alerts)

    def test_contact_event_is_none_in_restricted_mode(self):
        p = ThermalPipeline(restricted=True, contact_detector=_AlwaysContactDetector())
        r = p.process(*_triplet())
        assert r.contact_event is None

    def test_restricted_cooldown(self):
        p = ThermalPipeline(
            restricted=True,
            human_detector=_OneDetection(),
            restricted_area_cooldown_s=10.0,
        )
        r1 = p.process(*_triplet(ts=0.0))
        r2 = p.process(*_triplet(ts=5.0))    # within cooldown
        r3 = p.process(*_triplet(ts=11.0))   # after cooldown
        assert r1.restricted_area_alarm
        assert not r2.restricted_area_alarm
        assert r3.restricted_area_alarm

    def test_repr_shows_restricted(self):
        p = ThermalPipeline(restricted=True)
        assert "restricted=True" in repr(p)

    def test_restricted_false_runs_full_pipeline(self):
        # With restricted=False (default), fire and contact still run normally
        p = ThermalPipeline(
            restricted=False,
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
        )
        r = p.process(*_triplet())
        assert r.fire_alarm
        assert r.contact_alarm
        assert not r.restricted_area_alarm


class TestBothPaths:
    def test_fire_and_contact_both_alert(self):
        p = ThermalPipeline(
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
        )
        r = p.process(*_triplet())
        types = {a.type for a in r.alerts}
        assert AlertType.FIRE in types
        assert AlertType.CONTACT in types

    def test_two_alerts_in_result(self):
        p = ThermalPipeline(
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
        )
        r = p.process(*_triplet())
        assert len(r.alerts) == 2


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_fire_cooldown(self):
        p = ThermalPipeline(fire_detector=_AlarmFireDetector(), fire_cooldown_s=60.0)
        p.process(*_triplet(ts=0.0))   # fires alert, sets cooldown
        r = p.process(*_triplet(ts=5.0))
        assert not r.fire_alarm         # suppressed by cooldown
        p.reset()
        r_after = p.process(*_triplet(ts=6.0))
        assert r_after.fire_alarm       # cooldown cleared → fires again

    def test_reset_calls_detector_reset(self):
        reset_count = [0]

        class _TrackingFireDetector:
            def predict(self, f):
                return FireAlert(level=FireLevel.SAFE, timestamp=f.timestamp)
            def reset(self):
                reset_count[0] += 1

        p = ThermalPipeline(fire_detector=_TrackingFireDetector())
        p.reset()
        assert reset_count[0] == 1

    def test_reset_cooldown_specific_type(self):
        p = ThermalPipeline(
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
            fire_cooldown_s=60.0,
            contact_cooldown_s=60.0,
        )
        p.process(*_triplet(ts=0.0))
        p.reset_cooldown(AlertType.FIRE)

        r = p.process(*_triplet(ts=5.0))
        assert r.fire_alarm             # fire cooldown was cleared
        assert not r.contact_alarm      # contact cooldown still active

    def test_reset_cooldown_all_types(self):
        p = ThermalPipeline(
            fire_detector=_AlarmFireDetector(),
            contact_detector=_AlwaysContactDetector(),
            fire_cooldown_s=60.0,
            contact_cooldown_s=60.0,
        )
        p.process(*_triplet(ts=0.0))
        p.reset_cooldown()              # clear all

        r = p.process(*_triplet(ts=1.0))
        assert r.fire_alarm
        assert r.contact_alarm
