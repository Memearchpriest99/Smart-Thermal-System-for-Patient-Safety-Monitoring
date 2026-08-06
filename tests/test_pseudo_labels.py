"""Tests for pseudo (silver) person-detection label generation/consumption."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.training.datasets import DatasetIndex
from thermal_algorithms.training.pseudo_labels import (
    PseudoLabeledFrameDataset,
    generate_pseudo_person_labels,
    load_pseudo_labels,
    merge_by_session,
    save_pseudo_labels,
)


def _write_npz(path: Path, n_frames: int, profile=MLX90640) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = profile.resolution
    frames = np.full((n_frames, h, w), 20.0, dtype=np.float32)
    np.savez_compressed(path, frames=frames)


def _make_scene(root: Path, scene: str, n_frames: int = 4, annotated_frame: int | None = None) -> None:
    scene_dir = root / scene
    for ch in (0, 1, 2):
        _write_npz(scene_dir / f"ch{ch}_raw_data.npz", n_frames)
    if annotated_frame is not None:
        frames_dir = scene_dir / "ch0_frames"
        frames_dir.mkdir(parents=True)
        (frames_dir / f"frame_{annotated_frame:05d}.txt").write_text("1 0.5 0.5 0.2 0.2\n")


class _FakeDetector:
    """Deterministic stand-in for a fitted person detector."""

    def predict(self, frame: Frame) -> list[Detection]:
        # "Detects" a person whenever the frame's camera_id is even, for a
        # simple, checkable pattern.
        if frame.camera_id % 2 == 0:
            return [Detection(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, class_id=1, camera_id=frame.camera_id)]
        return []


class TestGeneratePseudoPersonLabels:
    def test_generates_boxes_per_channel_frame(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=3)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        assert "empty_room" in labels
        # All 3 channels get an entry per frame (explicit "checked, found
        # nothing" for cam 1 per _FakeDetector, not omitted).
        assert 0 in labels["empty_room"]
        assert 2 in labels["empty_room"]
        assert 1 in labels["empty_room"]
        assert all(boxes == [] for boxes in labels["empty_room"][1].values())
        assert len(labels["empty_room"][0]) == 3  # one entry per frame
        box = labels["empty_room"][0][0][0]
        assert box == (1.0, 2.0, 3.0, 4.0, 0.9)

    def test_skips_already_annotated_frames_by_default(self, tmp_path):
        _make_scene(tmp_path, "1_man_run", n_frames=3, annotated_frame=1)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        ch0_frames = labels["1_man_run"][0]
        assert 1 not in ch0_frames  # already has a real annotation
        assert 0 in ch0_frames and 2 in ch0_frames

    def test_does_not_skip_when_disabled(self, tmp_path):
        _make_scene(tmp_path, "1_man_run", n_frames=3, annotated_frame=1)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector(), skip_annotated=False)
        assert 1 in labels["1_man_run"][0]


class TestSaveLoadRoundtrip:
    def test_roundtrips_through_json(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=2)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        out = tmp_path / "pseudo.json"
        save_pseudo_labels(labels, out)
        loaded = load_pseudo_labels(out)

        assert loaded.keys() == labels.keys()
        assert loaded["empty_room"][0][0] == [(1.0, 2.0, 3.0, 4.0, 0.9)]


class TestPseudoLabeledFrameDataset:
    def test_yields_frames_with_silver_detections(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=3)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        ds = PseudoLabeledFrameDataset(index, labels)
        examples = list(ds)
        assert len(examples) == 3 * 3  # all 3 channels, 3 frames each
        even_cam_examples = [(f, d) for f, d in examples if f.camera_id % 2 == 0]
        odd_cam_examples = [(f, d) for f, d in examples if f.camera_id % 2 == 1]
        assert len(even_cam_examples) == 3 + 3  # cam 0 and cam 2
        assert len(odd_cam_examples) == 3  # cam 1
        for _frame, dets in even_cam_examples:
            assert len(dets) == 1
            assert dets[0].class_id == 1  # PERSON_CLASS_ID
            assert dets[0].score == 0.9
        for _frame, dets in odd_cam_examples:
            assert dets == []

    def test_min_score_filters_detections(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=2)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        ds = PseudoLabeledFrameDataset(index, labels, min_score=0.95)
        examples = list(ds)
        assert all(len(dets) == 0 for _frame, dets in examples)

    def test_by_session_groups_correctly(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=2)
        _make_scene(tmp_path, "heater_in_middle", n_frames=2)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        ds = PseudoLabeledFrameDataset(index, labels)
        groups = list(ds.by_session())
        assert [s.scene for s, _ex in groups] == ["empty_room", "heater_in_middle"]

    def test_empty_when_no_pseudo_labels_for_index(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=2)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        ds = PseudoLabeledFrameDataset(index, {})
        assert list(ds) == []
        assert list(ds.by_session()) == []


class TestMergeBySession:
    def test_chains_multiple_datasets(self, tmp_path):
        _make_scene(tmp_path, "empty_room", n_frames=2)
        _make_scene(tmp_path, "heater_in_middle", n_frames=2)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        labels = generate_pseudo_person_labels(index, _FakeDetector())

        ds_a = PseudoLabeledFrameDataset(index, {"empty_room": labels["empty_room"]})
        ds_b = PseudoLabeledFrameDataset(index, {"heater_in_middle": labels["heater_in_middle"]})

        scenes = [s.scene for s, _ex in merge_by_session(ds_a, ds_b)]
        assert scenes == ["empty_room", "heater_in_middle"]
