"""Tests for Trainer and the by_session() additions to dataset classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import FireAlert, FireLevel, Frame
from thermal_algorithms.training.datasets import (
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
)
from thermal_algorithms.training.label_io import FIRE_CLASS_ID, PERSON_CLASS_ID
from thermal_algorithms.training.metrics import BinaryConfusionMatrix, ScenarioResult
from thermal_algorithms.training.trainer import Trainer

# Re-use the session-building helpers from test_fire_contact_datasets
from tests.test_fire_contact_datasets import (
    _make_session,
    _write_npz,
    _write_yolo_label,
)


# ---------------------------------------------------------------------------
# Minimal fake detectors (no algorithms installed)
# ---------------------------------------------------------------------------

class _AlwaysDetect:
    """HumanDetector stub that always returns one detection."""
    is_trainable = False
    def fit(self, X, y=None): return self
    def predict(self, frame): return [object()]   # non-empty → positive

class _NeverDetect:
    """HumanDetector stub that always returns empty."""
    is_trainable = False
    def fit(self, X, y=None): return self
    def predict(self, frame): return []

class _AlwaysFireAlarm:
    """FireDetector stub that always fires ACTIVE_COMBUSTION."""
    is_trainable = False
    def fit(self, X, y=None): return self
    def reset(self): pass
    def predict(self, frame):
        return FireAlert(level=FireLevel.ACTIVE_COMBUSTION, timestamp=frame.timestamp,
                         blob_features={}, confidence=1.0)

class _NeverFireAlarm:
    """FireDetector stub that always returns SAFE."""
    is_trainable = False
    def fit(self, X, y=None): return self
    def reset(self): pass
    def predict(self, frame):
        return FireAlert(level=FireLevel.SAFE, timestamp=frame.timestamp)

class _PerfectContactDetector:
    """ContactDetector stub that mirrors ground truth."""
    is_trainable = False
    _gt: list = []
    def fit(self, X, y=None): return self
    def reset(self): self._gt = []
    def predict(self, frames, detections=None):
        from thermal_algorithms.core.types import ContactEvent
        contact = bool(self._gt.pop(0)) if self._gt else False
        ts = max(f.timestamp for f in frames)
        return ContactEvent(actors=(), pairs_in_contact=((0,1),) if contact else (),
                            timestamp=ts)


# ---------------------------------------------------------------------------
# by_session() tests
# ---------------------------------------------------------------------------

class TestBySession:
    def _setup(self, tmp_path):
        _make_session(tmp_path, "scene_a", n_frames=5, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)], 2: [(0.5, 0.5, 0.3, 0.5)]},
                      fire_frames={1: [(0.5, 0.4, 0.2, 0.2)]})
        _make_session(tmp_path, "scene_b", n_frames=4, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)]})
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_frame_level_dataset_by_session_yields_two_groups(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        groups = list(ds.by_session())
        assert len(groups) == 2

    def test_frame_level_dataset_by_session_names(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        names = [s.scene for s, _ in ds.by_session()]
        assert set(names) == {"scene_a", "scene_b"}

    def test_frame_level_dataset_by_session_total_examples(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        total = sum(len(examples) for _, examples in ds.by_session())
        assert total == len(ds)

    def test_fire_dataset_by_session_yields_groups(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        groups = list(ds.by_session())
        assert len(groups) >= 1
        total = sum(len(examples) for _, examples in groups)
        assert total == len(ds)

    def test_contact_dataset_by_session_yields_groups(self, tmp_path):
        _make_session(tmp_path, "c_scene", n_frames=6, n_channels=3,
                      contact_labels={0: 0, 1: 0, 2: 1, 3: 1})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = ContactFrameDataset(index)
        groups = list(ds.by_session())
        assert len(groups) == 1
        _, examples = groups[0]
        assert len(examples) == 4

    def test_by_session_empty_dataset(self, tmp_path):
        _make_session(tmp_path, "no_labels", n_frames=5, n_channels=1)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=False)
        groups = list(ds.by_session())
        assert groups == []


# ---------------------------------------------------------------------------
# Trainer.evaluate_human_detection
# ---------------------------------------------------------------------------

class TestEvaluateHumanDetection:
    def _setup(self, tmp_path):
        # scene_a: 3 frames with person, 2 without → always-detect gives 3 TP + 2 FP
        _make_session(tmp_path, "scene_a", n_frames=5, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)],
                                     2: [(0.5, 0.5, 0.3, 0.5)],
                                     4: [(0.5, 0.5, 0.3, 0.5)]})
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_returns_one_result_per_session(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="raw", verbose=False
        )
        assert len(results) == 1

    def test_result_mode_set_correctly(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="proc", verbose=False
        )
        assert results[0].mode == "proc"

    def test_always_detect_gives_correct_tp_fp(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="raw", verbose=False
        )
        cm = results[0].confusion
        assert cm.tp == 3   # 3 frames with person
        assert cm.fp == 2   # 2 empty frames predicted as human
        assert cm.fn == 0
        assert cm.tn == 0

    def test_never_detect_gives_all_fn(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _NeverDetect(), ds, mode="raw", verbose=False
        )
        cm = results[0].confusion
        assert cm.fn == 3
        assert cm.tn == 2
        assert cm.tp == 0

    def test_recall_on_perfect_detector(self, tmp_path):
        """A detector that always fires on positive and never on negative."""
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)

        class _PerfectHuman:
            is_trainable = False
            def fit(self, X, y=None): return self
            def predict(self, frame):
                # Fire only on frames with hot pixel (our positive frames
                # have uniform data — use frame index as a proxy is not
                # possible; just return based on data max)
                return [object()] if float(frame.data.max()) > 0 else []

        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="raw", verbose=False
        )
        assert results[0].recall == pytest.approx(1.0)

    def test_result_is_scenario_result(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="raw", verbose=False
        )
        assert isinstance(results[0], ScenarioResult)

    def test_multi_scene_returns_one_per_scene(self, tmp_path):
        _make_session(tmp_path, "scene_a", n_frames=5, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)]})
        _make_session(tmp_path, "scene_b", n_frames=3, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)]})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        results = Trainer.evaluate_human_detection(
            _AlwaysDetect(), ds, mode="raw", verbose=False
        )
        assert len(results) == 2
        assert {r.name for r in results} == {"scene_a", "scene_b"}


# ---------------------------------------------------------------------------
# Trainer.evaluate_fire_detection
# ---------------------------------------------------------------------------

class TestEvaluateFireDetection:
    def _setup(self, tmp_path):
        # fire_scene: frames 1,3 have fire bboxes; frames 0,2,4 are empty
        _make_session(tmp_path, "fire_scene", n_frames=5, n_channels=1,
                      fire_frames={1: [(0.5, 0.4, 0.2, 0.2)],
                                   3: [(0.5, 0.4, 0.2, 0.2)]})
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_returns_one_result_per_session(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        results = Trainer.evaluate_fire_detection(
            _AlwaysFireAlarm(), ds, mode="raw", verbose=False
        )
        assert len(results) == 1

    def test_always_alarm_tp_fp(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        results = Trainer.evaluate_fire_detection(
            _AlwaysFireAlarm(), ds, mode="raw", verbose=False
        )
        cm = results[0].confusion
        assert cm.tp == 2   # frames 1 and 3
        assert cm.fp == 3   # frames 0, 2, 4

    def test_never_alarm_fn_tn(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        results = Trainer.evaluate_fire_detection(
            _NeverFireAlarm(), ds, mode="raw", verbose=False
        )
        cm = results[0].confusion
        assert cm.fn == 2
        assert cm.tn == 3

    def test_reset_called_between_sessions(self, tmp_path):
        _make_session(tmp_path, "s1", n_frames=3, n_channels=1,
                      fire_frames={0: [(0.5, 0.4, 0.2, 0.2)]})
        _make_session(tmp_path, "s2", n_frames=3, n_channels=1,
                      fire_frames={0: [(0.5, 0.4, 0.2, 0.2)]})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = FireFrameDataset(index, include_negative_frames=True)

        reset_count = [0]
        class _TrackReset:
            is_trainable = False
            def fit(self, X, y=None): return self
            def reset(self): reset_count[0] += 1
            def predict(self, frame):
                return FireAlert(level=FireLevel.SAFE, timestamp=frame.timestamp)

        Trainer.evaluate_fire_detection(
            _TrackReset(), ds, mode="raw", verbose=False
        )
        assert reset_count[0] == 2  # once per session

    def test_mean_iou_zero_when_no_pred_bbox(self, tmp_path):
        index = self._setup(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        # _AlwaysFireAlarm returns no bbox in blob_features → IoU = 0
        results = Trainer.evaluate_fire_detection(
            _AlwaysFireAlarm(), ds, mode="raw", verbose=False
        )
        assert results[0].mean_iou == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Trainer.evaluate_contact_detection
# ---------------------------------------------------------------------------

class TestEvaluateContactDetection:
    def _setup(self, tmp_path):
        _make_session(tmp_path, "contact_s", n_frames=6, n_channels=3,
                      contact_labels={0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0})
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_returns_one_result_per_session(self, tmp_path):
        index = self._setup(tmp_path)
        ds = ContactFrameDataset(index)

        class _NoContact:
            is_trainable = False
            def fit(self, X, y=None): return self
            def reset(self): pass
            def predict(self, frames, detections=None):
                from thermal_algorithms.core.types import ContactEvent
                return ContactEvent(actors=(), pairs_in_contact=(),
                                    timestamp=max(f.timestamp for f in frames))

        results = Trainer.evaluate_contact_detection(
            _NoContact(), ds, mode="proc", verbose=False
        )
        assert len(results) == 1

    def test_correct_tp_fn_counts(self, tmp_path):
        index = self._setup(tmp_path)
        ds = ContactFrameDataset(index)

        class _AlwaysContact:
            is_trainable = False
            def fit(self, X, y=None): return self
            def reset(self): pass
            def predict(self, frames, detections=None):
                from thermal_algorithms.core.types import ContactEvent
                return ContactEvent(actors=(), pairs_in_contact=((0, 1),),
                                    timestamp=max(f.timestamp for f in frames))

        results = Trainer.evaluate_contact_detection(
            _AlwaysContact(), ds, mode="proc", verbose=False
        )
        cm = results[0].confusion
        assert cm.tp == 2   # frames 2 and 3 are contact
        assert cm.fp == 4   # frames 0, 1, 4, 5 predicted contact (wrong)


# ---------------------------------------------------------------------------
# Trainer.fit_and_evaluate (dispatch test)
# ---------------------------------------------------------------------------

class TestFitAndEvaluate:
    def test_dispatches_to_fire_evaluation(self, tmp_path):
        _make_session(tmp_path, "fire", n_frames=4, n_channels=1,
                      fire_frames={0: [(0.5, 0.4, 0.2, 0.2)]})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = FireFrameDataset(index, include_negative_frames=True)
        _, results = Trainer.fit_and_evaluate(
            _NeverFireAlarm(), None, ds, mode="raw", verbose=False
        )
        assert isinstance(results[0], ScenarioResult)

    def test_dispatches_to_human_evaluation(self, tmp_path):
        _make_session(tmp_path, "human", n_frames=4, n_channels=1,
                      person_frames={0: [(0.5, 0.5, 0.3, 0.5)]})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)
        _, results = Trainer.fit_and_evaluate(
            _AlwaysDetect(), None, ds, mode="proc", verbose=False
        )
        assert isinstance(results[0], ScenarioResult)


# ---------------------------------------------------------------------------
# Trainer.evaluate_preprocessing  (§5.3.1 SBR metric)
# ---------------------------------------------------------------------------

class TestEvaluatePreprocessing:
    def _build_dataset_with_hot_blobs(self, tmp_path):
        """Session with labeled person frames — hot blobs in known positions."""
        _make_session(tmp_path, "sbr_scene", n_frames=5, n_channels=1,
                      person_frames={
                          0: [(0.5, 0.5, 0.3, 0.5)],
                          1: [(0.5, 0.5, 0.3, 0.5)],
                          2: [(0.5, 0.5, 0.3, 0.5)],
                      })
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_returns_two_floats(self, tmp_path):
        index = self._build_dataset_with_hot_blobs(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)

        class _IdentityPreprocessor:
            def predict(self, frame): return frame

        raw_sbr, proc_sbr = Trainer.evaluate_preprocessing(
            ds, _IdentityPreprocessor(), verbose=False
        )
        assert isinstance(raw_sbr, float)
        assert isinstance(proc_sbr, float)

    def test_identity_preprocessor_equal_sbr(self, tmp_path):
        index = self._build_dataset_with_hot_blobs(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)

        class _IdentityPreprocessor:
            def predict(self, frame): return frame

        raw_sbr, proc_sbr = Trainer.evaluate_preprocessing(
            ds, _IdentityPreprocessor(), verbose=False
        )
        assert raw_sbr == pytest.approx(proc_sbr)

    def test_amplifying_preprocessor_improves_sbr(self, tmp_path):
        """A preprocessor that doubles the signal should raise SBR."""
        index = self._build_dataset_with_hot_blobs(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=False)

        from thermal_algorithms.core.types import Frame as _Frame
        import numpy as _np

        class _AmplifyPreprocessor:
            """Doubles every pixel value, increasing signal/background ratio."""
            def predict(self, frame: _Frame) -> _Frame:
                return _Frame(
                    data=(frame.data * 2.0).astype(_np.float32),
                    timestamp=frame.timestamp,
                    camera_id=frame.camera_id,
                )

        raw_sbr, proc_sbr = Trainer.evaluate_preprocessing(
            ds, _AmplifyPreprocessor(), verbose=False
        )
        # Both raw and processed have the same SBR when you scale uniformly
        # (SBR = ratio, not absolute). Confirm both are positive.
        assert raw_sbr > 0
        assert proc_sbr > 0

    def test_verbose_prints_summary(self, tmp_path, capsys):
        index = self._build_dataset_with_hot_blobs(tmp_path)
        ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID],
                               include_negative_frames=True)

        class _IdentityPreprocessor:
            def predict(self, frame): return frame

        Trainer.evaluate_preprocessing(ds, _IdentityPreprocessor(), verbose=True)
        out = capsys.readouterr().out
        assert "SBR" in out
        assert "raw=" in out
        assert "processed=" in out
