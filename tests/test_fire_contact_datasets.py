"""Tests for FireFrameDataset and ContactFrameDataset.

Both classes depend on the on-disk session layout, so tests build synthetic
session directories in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import ContactEvent, FireAlert, FireLevel, Frame
from thermal_algorithms.training.datasets import (
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
)
from thermal_algorithms.training.label_io import (
    FIRE_CLASS_ID,
    PERSON_CLASS_ID,
    save_contact_labels,
)


# ---------------------------------------------------------------------------
# Helpers — build a minimal on-disk session
# ---------------------------------------------------------------------------

def _write_npz(path: Path, n_frames: int, profile=MLX90640) -> None:
    w, h = profile.resolution
    rng = np.random.default_rng(0)
    frames = rng.standard_normal((n_frames, h, w)).astype(np.float32) + 25.0
    np.savez_compressed(path, frames=frames)


def _write_yolo_label(path: Path, bboxes: list[tuple[int, float, float, float, float]]) -> None:
    """Write a YOLO .txt label file.  Each bbox: (class, cx, cy, w, h) normalized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}" for cls, cx, cy, w, h in bboxes]
    path.write_text("\n".join(lines) + "\n" if lines else "")


def _make_session(
    root: Path,
    scene: str,
    n_frames: int = 10,
    n_channels: int = 3,
    *,
    fire_frames: dict[int, list] | None = None,
    person_frames: dict[int, list] | None = None,
    contact_labels: dict[int, int] | None = None,
    profile=MLX90640,
) -> Path:
    """Create a minimal session directory for testing.

    Args:
        root: Dataset root.
        scene: Scene folder name.
        n_frames: Number of frames in each channel's npz.
        n_channels: How many channel npz files to create (0, 1, 2, ...).
        fire_frames: {frame_idx: [(cx,cy,w,h), ...]} fire bboxes (normalized).
        person_frames: {frame_idx: [(cx,cy,w,h), ...]} person bboxes (normalized).
        contact_labels: {frame_idx: 0/1}.
        profile: Sensor profile.
    """
    w, h = profile.resolution
    session_dir = root / scene
    session_dir.mkdir(parents=True, exist_ok=True)

    for ch in range(n_channels):
        _write_npz(session_dir / f"ch{ch}_raw_data.npz", n_frames, profile)

    (session_dir / "classes.txt").write_text("fire\nperson\n")

    # Write YOLO label files
    all_labeled_frames: set[int] = set()
    if fire_frames:
        all_labeled_frames |= set(fire_frames)
    if person_frames:
        all_labeled_frames |= set(person_frames)

    for ch in range(n_channels):
        if fire_frames or person_frames:
            frames_dir = session_dir / f"ch{ch}_frames"
            frames_dir.mkdir(exist_ok=True)

            for frame_idx in all_labeled_frames:
                label_path = frames_dir / f"frame_{frame_idx:05d}.txt"
                bboxes = []
                if fire_frames and frame_idx in fire_frames:
                    bboxes += [(FIRE_CLASS_ID, cx, cy, bw, bh)
                               for cx, cy, bw, bh in fire_frames[frame_idx]]
                if person_frames and frame_idx in person_frames:
                    bboxes += [(PERSON_CLASS_ID, cx, cy, bw, bh)
                               for cx, cy, bw, bh in person_frames[frame_idx]]
                _write_yolo_label(label_path, bboxes)

    if contact_labels:
        save_contact_labels(contact_labels, session_dir / "contact_labels.csv")

    return session_dir


# ---------------------------------------------------------------------------
# FireFrameDataset
# ---------------------------------------------------------------------------

class TestFireFrameDataset:
    def _build_index(self, tmp_path) -> DatasetIndex:
        _make_session(
            tmp_path, "fire_scene",
            n_frames=8,
            fire_frames={2: [(0.5, 0.4, 0.2, 0.2)], 5: [(0.6, 0.5, 0.15, 0.15)]},
            person_frames={1: [(0.5, 0.5, 0.3, 0.5)], 2: [(0.5, 0.5, 0.3, 0.5)]},
        )
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_fire_frames_have_active_combustion_level(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index)
        alerts = [alert for _, alert in ds]
        fire_alerts = [a for a in alerts if a.level == FireLevel.ACTIVE_COMBUSTION]
        assert len(fire_alerts) >= 1

    def test_negative_frames_have_safe_level(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        alerts = [alert for _, alert in ds]
        # At least person-only frames should be SAFE in the fire dataset
        safe_alerts = [a for a in alerts if a.level == FireLevel.SAFE]
        assert len(safe_alerts) >= 1

    def test_fire_alert_has_bbox_in_blob_features(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index)
        fire_items = [(f, a) for f, a in ds if a.level == FireLevel.ACTIVE_COMBUSTION]
        assert fire_items, "expected at least one fire frame"
        _, alert = fire_items[0]
        assert "bboxes" in alert.blob_features
        assert len(alert.blob_features["bboxes"]) >= 1

    def test_frame_is_frame_type(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index)
        frame, alert = ds[0]
        assert isinstance(frame, Frame)
        assert isinstance(alert, FireAlert)

    def test_len_matches_iteration_count(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index)
        assert len(ds) == sum(1 for _ in ds)

    def test_exclude_negatives(self, tmp_path):
        index = self._build_index(tmp_path)
        ds_all = FireFrameDataset(index, include_negative_frames=True)
        ds_pos = FireFrameDataset(index, include_negative_frames=False)
        assert len(ds_pos) <= len(ds_all)
        # All items in positive-only dataset must be fire frames
        for _, alert in ds_pos:
            assert alert.level == FireLevel.ACTIVE_COMBUSTION

    def test_class_filter_excludes_person_bboxes(self, tmp_path):
        """Fire dataset must not produce person detections."""
        # Frame 1 has only a person bbox (no fire); should appear as SAFE
        index = self._build_index(tmp_path)
        ds = FireFrameDataset(index, include_negative_frames=True)
        person_only_alerts = [
            a for f, a in ds
            if f.metadata.get("session_id", "").startswith("fire_scene")
            and a.blob_features.get("n_fire_boxes", 0) == 0
        ]
        for a in person_only_alerts:
            assert a.level == FireLevel.SAFE

    def test_scenes_filter(self, tmp_path):
        _make_session(tmp_path, "other_scene", n_frames=5,
                      fire_frames={0: [(0.5, 0.4, 0.2, 0.2)]})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds_all = FireFrameDataset(index)
        ds_filtered = FireFrameDataset(index, scenes=["fire_scene"])
        # fire_scene is a subset
        assert len(ds_filtered) < len(ds_all) or len(ds_all) == 0


# ---------------------------------------------------------------------------
# ContactFrameDataset
# ---------------------------------------------------------------------------

class TestContactFrameDataset:
    def _build_index(self, tmp_path) -> DatasetIndex:
        _make_session(
            tmp_path, "contact_scene",
            n_frames=10, n_channels=3,
            contact_labels={0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0},
        )
        return DatasetIndex(tmp_path, sensor_profile=MLX90640)

    def test_yields_three_view_frames_and_event(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        frames, event = ds[0]
        assert len(frames) == 3
        assert all(isinstance(f, Frame) for f in frames)
        assert isinstance(event, ContactEvent)

    def test_contact_labels_propagated(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        contact_events = [e for _, e in ds if e.any_contact]
        no_contact_events = [e for _, e in ds if not e.any_contact]
        assert len(contact_events) == 2    # frames 2 and 3
        assert len(no_contact_events) == 4  # frames 0, 1, 4, 5

    def test_total_length(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        assert len(ds) == 6

    def test_label_counts(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        counts = ds.label_counts()
        assert counts["contact"] == 2
        assert counts["no_contact"] == 4

    def test_len_matches_iter(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        assert len(ds) == sum(1 for _ in ds)

    def test_session_without_csv_excluded(self, tmp_path):
        # One session has labels, one doesn't
        _make_session(tmp_path, "with_labels", n_frames=5, n_channels=3,
                      contact_labels={0: 1, 1: 0})
        _make_session(tmp_path, "no_labels", n_frames=5, n_channels=3)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = ContactFrameDataset(index)
        assert len(ds) == 2  # only the labeled session

    def test_session_with_fewer_than_3_channels_excluded(self, tmp_path):
        # Only 2 channels → should be skipped (contact needs 3-view)
        _make_session(tmp_path, "two_cam", n_frames=5, n_channels=2,
                      contact_labels={0: 1, 1: 0})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = ContactFrameDataset(index)
        assert len(ds) == 0

    def test_frame_idx_out_of_bounds_skipped(self, tmp_path):
        # Label references frame 999 which doesn't exist in a 10-frame session
        _make_session(tmp_path, "s", n_frames=10, n_channels=3,
                      contact_labels={0: 1, 999: 0})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds = ContactFrameDataset(index)
        assert len(ds) == 1  # only frame 0 is valid

    def test_scenes_filter(self, tmp_path):
        _make_session(tmp_path, "scene_a", n_frames=5, n_channels=3,
                      contact_labels={0: 1, 1: 0})
        _make_session(tmp_path, "scene_b", n_frames=5, n_channels=3,
                      contact_labels={0: 0, 1: 1, 2: 0})
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640)
        ds_a = ContactFrameDataset(index, scenes=["scene_a"])
        ds_b = ContactFrameDataset(index, scenes=["scene_b"])
        ds_all = ContactFrameDataset(index)
        assert len(ds_a) == 2
        assert len(ds_b) == 3
        assert len(ds_all) == 5

    def test_timestamps_derived_from_fps(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        _, event = ds[0]
        fps = index.sessions[0].fps
        # Frame 0 → timestamp = 0 / fps = 0.0
        assert event.timestamp == pytest.approx(0.0 / fps)

    def test_index_and_iter_match(self, tmp_path):
        index = self._build_index(tmp_path)
        ds = ContactFrameDataset(index)
        by_index = [ds[i] for i in range(len(ds))]
        by_iter = list(ds)
        for (f1, e1), (f2, e2) in zip(by_index, by_iter):
            assert e1.any_contact == e2.any_contact
            assert e1.timestamp == pytest.approx(e2.timestamp)
