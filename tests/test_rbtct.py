"""Tests for RBTCTDetector (Rule-Based Temporal Contact Tracker).

Two things are worth locking down here beyond the usual construction/persistence
checks:

* the per-camera state machine and the veto quorum, because those encode the
  whole detector (there are no learned weights to fall back on), and
* the causal attack/release gate, because it is the one place this class
  deliberately DIVERGES from the offline config-D morphology it was promoted
  from, and a regression there would silently change every reported number.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from thermal_algorithms.contact_detection.rbtct import (
    RBTCTDetector,
    _box_gap,
    _iou,
    _merge_oversegmented,
    _min_gap,
)
from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Detection, Frame

H, W = WAVESHARE_26984.height, WAVESHARE_26984.width


def _frame(arr, cam=0, ts=0.0):
    return Frame(data=arr.astype(np.float32), timestamp=ts, camera_id=cam)


def _det(bbox, cam=0):
    return Detection(bbox=bbox, score=0.9, class_id=1, camera_id=cam)


def _scene(*, blobs, background=0.0, noise=0.0, seed=0):
    """(H, W) residual with a hot rectangle per (x, y, w, h) in blobs."""
    rng = np.random.default_rng(seed)
    a = np.full((H, W), background, dtype=np.float32)
    if noise:
        a += rng.normal(0, noise, a.shape).astype(np.float32)
    for (x, y, w, h) in blobs:
        a[int(y):int(y + h), int(x):int(x + w)] = 10.0
    return a


class TestGeometryHelpers:
    def test_box_gap_zero_when_overlapping(self):
        assert _box_gap((0, 0, 10, 10), (5, 5, 10, 10)) == 0.0

    def test_box_gap_positive_when_separated(self):
        assert _box_gap((0, 0, 10, 10), (20, 0, 10, 10)) == pytest.approx(10.0)

    def test_min_gap_infinite_for_single_box(self):
        assert _min_gap([(0, 0, 5, 5)]) == math.inf

    def test_iou_identical_boxes_is_one(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_merge_collapses_contained_box(self):
        # A small box fully inside a big one is one person detected twice:
        # low IoU, containment 1.0 -> must merge (this is the case an
        # IoU-only rule gets wrong).
        merged = _merge_oversegmented([(0, 0, 20, 20), (5, 5, 4, 4)], 0.2, 0.7)
        assert len(merged) == 1

    def test_merge_keeps_genuinely_separate_boxes(self):
        merged = _merge_oversegmented([(0, 0, 10, 10), (40, 0, 10, 10)], 0.2, 0.7)
        assert len(merged) == 2


class TestConstruction:
    def test_requires_sensor_profile(self):
        with pytest.raises(ValueError):
            RBTCTDetector(sensor_profile=None)

    def test_is_not_trainable_and_usable_immediately(self):
        det = RBTCTDetector(WAVESHARE_26984)
        assert det.is_trainable is False
        assert det.is_fitted is True          # rule-based: no training needed

    def test_fit_is_a_noop_returning_self(self):
        det = RBTCTDetector(WAVESHARE_26984)
        assert det.fit([], []) is det


class TestCameraState:
    def test_no_boxes_is_clear(self):
        det = RBTCTDetector(WAVESHARE_26984)
        assert det._camera_state([], _scene(blobs=[])) == "C"

    def test_single_box_is_ambiguous_not_negative(self):
        # The blob-merge failure mode: one box may itself be two merged
        # people, so a single box must never count as evidence of NO contact.
        det = RBTCTDetector(WAVESHARE_26984)
        assert det._camera_state([(10, 10, 8, 12)], _scene(blobs=[(10, 10, 8, 12)])) == "M"

    def test_two_boxes_in_one_blob_is_touch(self):
        det = RBTCTDetector(WAVESHARE_26984)
        resid = _scene(blobs=[(10, 10, 20, 12)])          # ONE warm region
        boxes = [(11, 11, 8, 10), (21, 11, 8, 10)]        # TWO box centres in it
        assert det._camera_state(boxes, resid) == "T"

    def test_two_separated_boxes_far_apart_is_clear(self):
        det = RBTCTDetector(WAVESHARE_26984)
        resid = _scene(blobs=[(5, 10, 8, 10), (55, 10, 8, 10)])
        assert det._camera_state([(5, 10, 8, 10), (55, 10, 8, 10)], resid) == "C"

    def test_two_separated_but_close_boxes_is_near(self):
        det = RBTCTDetector(WAVESHARE_26984, tau_near_px=20.0)
        resid = _scene(blobs=[(5, 10, 8, 10), (18, 10, 8, 10)])
        assert det._camera_state([(5, 10, 8, 10), (18, 10, 8, 10)], resid) == "N"


class TestQuorum:
    def _predict_states(self, det, states):
        """Drive raw_decision() via crafted per-camera inputs producing the
        requested states."""
        frames, dets = [], []
        for c, s in enumerate(states):
            if s == "C":
                blobs = [(5, 10, 8, 10), (55, 10, 8, 10)]
                boxes = [(5, 10, 8, 10), (55, 10, 8, 10)]
            elif s == "N":
                # two separate warm bodies, close but not merged
                blobs = [(5, 10, 8, 10), (16, 10, 8, 10)]
                boxes = [(5, 10, 8, 10), (16, 10, 8, 10)]
            elif s == "M":
                blobs = [(10, 10, 8, 12)]
                boxes = [(10, 10, 8, 12)]
            elif s == "T":
                blobs = [(10, 10, 20, 12)]
                boxes = [(11, 11, 8, 10), (21, 11, 8, 10)]
            else:
                raise AssertionError(s)
            frames.append(_frame(_scene(blobs=blobs), cam=c))
            dets.append([_det(b, cam=c) for b in boxes])
        return det.raw_decision(tuple(frames), tuple(dets))

    def test_any_clear_camera_vetoes(self):
        det = RBTCTDetector(WAVESHARE_26984)
        # Two cameras see the merge, but a third cleanly resolves two bodies.
        assert self._predict_states(det, ["T", "T", "C"]) == 0

    def test_touch_plus_ambiguous_fires(self):
        det = RBTCTDetector(WAVESHARE_26984)
        assert self._predict_states(det, ["T", "M", "M"]) == 1

    def test_all_cameras_agreeing_fires(self):
        det = RBTCTDetector(WAVESHARE_26984)
        assert self._predict_states(det, ["T", "T", "T"]) == 1

    def test_lone_touch_without_corroboration_does_not_fire(self):
        det = RBTCTDetector(WAVESHARE_26984, tau_near_px=20.0)
        # One camera sees the merge; the others resolve two separate bodies
        # close enough not to veto ("N"). votes=1 but votes+merged=1 < 2, so
        # there is no corroborating camera -> must not fire.
        assert self._predict_states(det, ["T", "N", "N"]) == 0

    def test_ambiguous_only_never_fires(self):
        det = RBTCTDetector(WAVESHARE_26984)
        # No camera positively observes a merge.
        assert self._predict_states(det, ["M", "M", "M"]) == 0


class TestTemporalGate:
    """The causal attack/release gate -- see the module docstring for why this
    deliberately differs from the offline morphology of config-D."""

    def _run(self, det, raw):
        det.reset()
        return [det._gate(v) for v in raw]

    def test_attack_suppresses_isolated_blips(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=3, release_frames=0)
        assert self._run(det, [0, 1, 0, 0]) == [0, 0, 0, 0]

    def test_attack_asserts_after_n_consecutive(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=3, release_frames=0)
        assert self._run(det, [1, 1, 1, 1]) == [0, 0, 1, 1]

    def test_release_bridges_short_dropout(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=1, release_frames=3)
        # 1 asserts; the two 0s are held; recovery keeps it asserted.
        assert self._run(det, [1, 0, 0, 1, 0, 0, 0, 0, 0]) == [1, 1, 1, 1, 1, 1, 1, 0, 0]

    def test_release_zero_drops_immediately(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=1, release_frames=0)
        assert self._run(det, [1, 0, 1, 0]) == [1, 0, 1, 0]

    def test_gate_is_causal(self):
        """A future positive must never change a past output -- the property
        offline morphological closing violates and this gate preserves."""
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=1, release_frames=2)
        short = self._run(det, [0, 0, 1])
        long = self._run(det, [0, 0, 1, 1, 1, 1])
        assert long[: len(short)] == short

    def test_reset_clears_hold(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=1, release_frames=5)
        det._gate(1)
        det.reset()
        assert det._gate(0) == 0


class TestPredict:
    def test_predict_without_detections_is_negative(self):
        det = RBTCTDetector(WAVESHARE_26984)
        frames = tuple(_frame(_scene(blobs=[]), cam=c) for c in range(3))
        ev = det.predict(frames, None)
        assert ev.any_contact is False

    def test_predict_propagates_timestamp(self):
        det = RBTCTDetector(WAVESHARE_26984)
        frames = tuple(_frame(_scene(blobs=[]), cam=c, ts=4.25) for c in range(3))
        assert det.predict(frames, None).timestamp == pytest.approx(4.25)

    def test_predict_exposes_raw_decision_in_debug(self):
        det = RBTCTDetector(WAVESHARE_26984)
        frames = tuple(_frame(_scene(blobs=[]), cam=c) for c in range(3))
        assert "raw" in det.predict(frames, None).debug


class TestCalibration:
    def test_calibrate_recovers_a_release_that_bridges_gaps(self):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=1, release_frames=0)
        # Ground truth is a solid contact block; the raw stream has holes in it.
        raw = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
        lab = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
        a, r = det.calibrate_temporal([(raw, lab)])
        assert r >= 1, "should learn to bridge the dropouts"
        assert det.get_params()["release_frames"] == r   # persisted, not just live

    def test_calibrate_rejects_empty_input(self):
        det = RBTCTDetector(WAVESHARE_26984)
        with pytest.raises(ValueError):
            det.calibrate_temporal([])


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        det = RBTCTDetector(WAVESHARE_26984, attack_frames=4, release_frames=6, k=1.5)
        p = tmp_path / "rbtct.thalg"
        det.save(p)
        loaded = RBTCTDetector.load(p)
        assert loaded._attack == 4
        assert loaded._release == 6
        assert loaded._k == pytest.approx(1.5)
        assert loaded.get_params()["attack_frames"] == 4

    def test_profile_is_preserved(self, tmp_path):
        det = RBTCTDetector(MLX90640)
        p = tmp_path / "rbtct_mlx.thalg"
        det.save(p)
        assert RBTCTDetector.load(p).sensor_profile.name == MLX90640.name
