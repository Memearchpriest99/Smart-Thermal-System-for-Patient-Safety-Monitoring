"""Tests for training/label_io.py and training/datasets.py.

Most tests build a synthetic dataset on a tmp_path so they don't depend on
the real (large, evolving) recording set. A couple of tests at the bottom
exercise the real dataset when present, but skip cleanly when it isn't.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Detection
from thermal_algorithms.training.label_io import (
    frame_index_from_label_path,
    list_label_files,
    load_classes_file,
    load_yolo_labels,
)
from thermal_algorithms.training.datasets import (
    DatasetIndex,
    FrameLevelDataset,
    sample_background_patches,
)


# ---------------------------------------------------------------------------
# Fixtures: a synthetic on-disk dataset matching both real layouts
# ---------------------------------------------------------------------------

def _write_yolo(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_session(
    root: Path,
    *,
    n_frames: int,
    labeled_frames: dict[int, list[str]] | None = None,  # {frame_idx: [lines]} per channel
    channels: tuple[int, ...] = (0, 1, 2),
    write_classes_txt: bool = True,
    layout: str = "B",   # "A" = nested under timestamp_session, "B" = direct
    base_temp: float = 20.0,
) -> Path:
    """Materialize a synthetic recording session on disk. Returns the path
    actually written (which differs between layouts)."""
    rng = np.random.default_rng(42)
    if layout == "A":
        session = root / "20260101_000000"
    else:
        session = root
    session.mkdir(parents=True, exist_ok=True)

    for ch in channels:
        # Synthetic .npz with `frames` array of shape (n_frames, 24, 32)
        frames = rng.normal(base_temp, 0.5, size=(n_frames, 24, 32)).astype(np.float32)
        np.savez(session / f"ch{ch}_raw_data.npz", frames=frames)

        if labeled_frames:
            for fi, lines in labeled_frames.items():
                _write_yolo(session / f"ch{ch}_frames" / f"frame_{fi:05d}.txt", lines)

    if write_classes_txt:
        (session / "classes.txt").write_text("fire\nhuman\n")
    return session


# ---------------------------------------------------------------------------
# label_io — YOLO bbox parsing
# ---------------------------------------------------------------------------

class TestYoloLabels:
    def test_parses_single_box(self, tmp_path):
        p = tmp_path / "frame_00000.txt"
        p.write_text("1 0.5 0.5 0.25 0.5\n")
        dets = load_yolo_labels(p, frame_shape=(24, 32), camera_id=0)
        assert len(dets) == 1
        x, y, w, h = dets[0].bbox
        assert (x, y, w, h) == (12.0, 6.0, 8.0, 12.0)
        assert dets[0].class_id == 1
        assert dets[0].camera_id == 0

    def test_parses_multiple_boxes(self, tmp_path):
        p = tmp_path / "frame_00001.txt"
        p.write_text("0 0.1 0.1 0.1 0.1\n1 0.9 0.9 0.1 0.1\n")
        dets = load_yolo_labels(p, frame_shape=(24, 32))
        assert len(dets) == 2
        assert {d.class_id for d in dets} == {0, 1}

    def test_missing_file_yields_empty_list(self, tmp_path):
        assert load_yolo_labels(tmp_path / "nope.txt", frame_shape=(24, 32)) == []

    def test_class_filter(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.1 0.1\n")
        humans_only = load_yolo_labels(p, frame_shape=(24, 32), class_filter=[1])
        assert len(humans_only) == 1
        assert humans_only[0].class_id == 1

    def test_blank_lines_ignored(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("\n1 0.5 0.5 0.1 0.1\n\n")
        assert len(load_yolo_labels(p, frame_shape=(24, 32))) == 1

    def test_clamps_to_frame_extents(self, tmp_path):
        p = tmp_path / "f.txt"
        # cx=0.95, w=0.5 → right edge would go off-frame; we expect clamping
        p.write_text("1 0.95 0.5 0.5 0.5\n")
        dets = load_yolo_labels(p, frame_shape=(24, 32))
        x, y, w, h = dets[0].bbox
        assert x + w <= 32 + 1e-6
        assert y + h <= 24 + 1e-6

    def test_rejects_malformed_line(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("1 0.5 0.5 0.1\n")  # only 4 tokens
        with pytest.raises(ValueError, match="5 tokens"):
            load_yolo_labels(p, frame_shape=(24, 32))

    def test_rejects_non_integer_class(self, tmp_path):
        # Mirrors the real-world classes.txt edge case where 'fire'/'human'
        # appear in a stray classes.txt — our parser must not accept those
        # as box rows.
        p = tmp_path / "f.txt"
        p.write_text("fire 0.5 0.5 0.1 0.1\n")
        with pytest.raises(ValueError, match="class index"):
            load_yolo_labels(p, frame_shape=(24, 32))


class TestClassesFile:
    def test_loads_classes(self, tmp_path):
        p = tmp_path / "classes.txt"
        p.write_text("fire\nhuman\n")
        assert load_classes_file(p) == {0: "fire", 1: "human"}

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "classes.txt"
        p.write_text("fire\n\nhuman\n")
        assert load_classes_file(p) == {0: "fire", 1: "human"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_classes_file(tmp_path / "nope.txt")


class TestLabelHelpers:
    def test_list_label_files_sorted(self, tmp_path):
        for i in [3, 1, 2]:
            (tmp_path / f"frame_{i:05d}.txt").write_text("")
        files = list_label_files(tmp_path)
        indices = [frame_index_from_label_path(p) for p in files]
        assert indices == [1, 2, 3]

    def test_list_excludes_classes_txt(self, tmp_path):
        (tmp_path / "classes.txt").write_text("fire\nhuman\n")
        (tmp_path / "frame_00000.txt").write_text("")
        files = list_label_files(tmp_path)
        assert all(p.name.startswith("frame_") for p in files)


# ---------------------------------------------------------------------------
# DatasetIndex
# ---------------------------------------------------------------------------

class TestDatasetIndex:
    def test_empty_root(self, tmp_path):
        idx = DatasetIndex(tmp_path / "doesnotexist")
        assert idx.sessions == ()
        assert idx.labeled_sessions() == ()

    def test_finds_layout_a(self, tmp_path):
        scene = tmp_path / "scene_a"
        _write_session(scene, n_frames=10, layout="A")
        idx = DatasetIndex(tmp_path)
        assert len(idx.sessions) == 1
        s = idx.sessions[0]
        assert s.scene == "scene_a"
        assert s.session == "20260101_000000"

    def test_finds_layout_b(self, tmp_path):
        scene = tmp_path / "scene_b"
        _write_session(scene, n_frames=10, layout="B")
        idx = DatasetIndex(tmp_path)
        assert len(idx.sessions) == 1
        assert idx.sessions[0].session is None

    def test_labeled_sessions_filter(self, tmp_path):
        # One labeled, one unlabeled
        _write_session(tmp_path / "with_labels", n_frames=5,
                       labeled_frames={0: ["1 0.5 0.5 0.1 0.1"]}, layout="B")
        _write_session(tmp_path / "without_labels", n_frames=5, layout="B")
        idx = DatasetIndex(tmp_path)
        labeled = idx.labeled_sessions()
        assert len(labeled) == 1
        assert labeled[0].scene == "with_labels"

    def test_loads_classes_txt(self, tmp_path):
        _write_session(tmp_path / "scene", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        assert idx.sessions[0].class_names == {0: "fire", 1: "human"}

    def test_session_id(self, tmp_path):
        _write_session(tmp_path / "a", n_frames=3, layout="A")
        _write_session(tmp_path / "b", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        ids = sorted(s.session_id for s in idx.sessions)
        assert ids == ["a/20260101_000000", "b"]

    def test_find_by_scene(self, tmp_path):
        _write_session(tmp_path / "alpha", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        s = idx.find("alpha")
        assert s.scene == "alpha"
        with pytest.raises(KeyError):
            idx.find("nope")

    def test_n_frames_matches_npz(self, tmp_path):
        _write_session(tmp_path / "scene", n_frames=42, layout="B")
        idx = DatasetIndex(tmp_path)
        assert idx.sessions[0].n_frames == 42

    def test_empty_room_sessions(self, tmp_path):
        _write_session(tmp_path / "emptyroom_test", n_frames=3, layout="B")
        _write_session(tmp_path / "personwalk", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        empty = idx.empty_room_sessions()
        assert len(empty) == 1
        assert empty[0].scene == "emptyroom_test"


# ---------------------------------------------------------------------------
# FrameLevelDataset
# ---------------------------------------------------------------------------

class TestFrameLevelDataset:
    @pytest.fixture
    def two_sessions(self, tmp_path):
        # Session A: layout B, 5 frames, frames 0,1,3 labeled with a human box
        _write_session(
            tmp_path / "scene_one", n_frames=5, layout="B",
            labeled_frames={
                0: ["1 0.5 0.5 0.25 0.5"],
                1: ["1 0.6 0.6 0.25 0.5"],
                3: ["1 0.5 0.5 0.25 0.5", "0 0.1 0.1 0.05 0.05"],  # human + fire
            },
        )
        # Session B: layout A, no labels (excluded by default)
        _write_session(
            tmp_path / "scene_two", n_frames=4, layout="A",
        )
        return DatasetIndex(tmp_path)

    def test_len(self, two_sessions):
        ds = FrameLevelDataset(two_sessions)
        # scene_one: 3 labeled frames × 3 channels = 9
        assert len(ds) == 9

    def test_yields_frame_and_detections(self, two_sessions):
        ds = FrameLevelDataset(two_sessions)
        frame, dets = ds[0]
        assert frame.shape == (24, 32)
        assert all(isinstance(d, Detection) for d in dets)

    def test_class_filter_humans_only(self, two_sessions):
        ds = FrameLevelDataset(two_sessions, class_filter=[1])
        for _, dets in ds:
            for d in dets:
                assert d.class_id == 1

    def test_class_filter_fires_only(self, two_sessions):
        ds = FrameLevelDataset(two_sessions, class_filter=[0])
        # Only frame 3 has a fire box → 1 fire × 3 channels = 3 frames with a fire detection
        total_fires = sum(len(d) for _, d in ds)
        assert total_fires == 3

    def test_camera_id_propagated(self, two_sessions):
        ds = FrameLevelDataset(two_sessions, channels=(2,))
        for frame, dets in ds:
            assert frame.camera_id == 2
            for d in dets:
                assert d.camera_id == 2

    def test_include_negative_frames(self, two_sessions):
        ds = FrameLevelDataset(two_sessions, include_negative_frames=True)
        # scene_one: 5 frames × 3 channels = 15
        assert len(ds) == 15
        # Some yielded examples should have empty detection lists
        empties = sum(1 for _, d in ds if not d)
        assert empties > 0

    def test_scene_filter(self, tmp_path):
        _write_session(tmp_path / "alpha", n_frames=3, layout="B",
                       labeled_frames={0: ["1 0.5 0.5 0.1 0.1"]})
        _write_session(tmp_path / "beta", n_frames=3, layout="B",
                       labeled_frames={0: ["1 0.5 0.5 0.1 0.1"]})
        idx = DatasetIndex(tmp_path)
        ds = FrameLevelDataset(idx, scenes=["alpha"])
        for frame, _ in ds:
            assert frame.metadata["session_id"].startswith("alpha")


# ---------------------------------------------------------------------------
# Background patch sampling
# ---------------------------------------------------------------------------

class TestBackgroundPatches:
    def test_samples_correct_shape(self, tmp_path):
        _write_session(tmp_path / "emptyroom_x", n_frames=5, layout="B")
        idx = DatasetIndex(tmp_path)
        patches = sample_background_patches(idx, patch_h=8, patch_w=8, n_patches=12)
        assert patches.shape == (12, 8, 8)
        assert patches.dtype == np.float32

    def test_reproducible_with_seed(self, tmp_path):
        _write_session(tmp_path / "emptyroom_x", n_frames=5, layout="B")
        idx = DatasetIndex(tmp_path)
        a = sample_background_patches(idx, patch_h=4, patch_w=4, n_patches=10,
                                       rng=np.random.default_rng(0))
        b = sample_background_patches(idx, patch_h=4, patch_w=4, n_patches=10,
                                       rng=np.random.default_rng(0))
        np.testing.assert_array_equal(a, b)

    def test_rejects_oversize_patch(self, tmp_path):
        _write_session(tmp_path / "emptyroom_x", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        with pytest.raises(ValueError, match="smaller than"):
            sample_background_patches(idx, patch_h=50, patch_w=50, n_patches=1)

    def test_no_empty_rooms_raises(self, tmp_path):
        _write_session(tmp_path / "personwalk", n_frames=3, layout="B")
        idx = DatasetIndex(tmp_path)
        with pytest.raises(ValueError, match="No empty-room"):
            sample_background_patches(idx, patch_h=4, patch_w=4, n_patches=1)


# ---------------------------------------------------------------------------
# Smoke test on the real dataset (skipped if not available)
# ---------------------------------------------------------------------------

REAL_DATASET = Path("/sessions/magical-youthful-euler/mnt/dataset")


@pytest.mark.skipif(not REAL_DATASET.is_dir(), reason="real dataset not mounted")
class TestRealDataset:
    def test_index_finds_known_labeled_sessions(self):
        idx = DatasetIndex(REAL_DATASET)
        labeled_scenes = {s.scene for s in idx.labeled_sessions()}
        assert "2pplfight" in labeled_scenes
        assert "2pplwithtouch" in labeled_scenes

    def test_framelevel_dataset_yields_humans(self):
        idx = DatasetIndex(REAL_DATASET)
        ds = FrameLevelDataset(idx, class_filter=[1])
        assert len(ds) > 0
        # Spot-check the first 5 examples
        for i in range(min(5, len(ds))):
            frame, dets = ds[i]
            assert frame.shape == (24, 32)
            for d in dets:
                assert d.class_id == 1
                x, y, w, h = d.bbox
                assert 0 <= x and x + w <= 32 + 1e-3
                assert 0 <= y and y + h <= 24 + 1e-3
