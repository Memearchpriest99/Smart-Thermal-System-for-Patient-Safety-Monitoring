"""Tests for the multi-source (waveshare + synth/room-1) dataset unification."""

import datetime as dt

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
import hdf5plugin  # noqa: E402,F401  registers the Zstd filter; import after skip-guard

from thermal_algorithms.core.types import FireLevel
from thermal_algorithms.training.multi_source import (
    HDF5Session,
    HDF5SessionRef,
    MultiSourceContactDataset,
    MultiSourceFireDataset,
    discover_hdf5_sessions,
)

_EPOCH = 1_700_000_000.0
CSV_HEADER = "Room_ID,Date,Start_Time,End_Time,Event_Class,Event_Class_ID,Cameras,Timestamp\n"


def _write_chunk(path, *, n, start_epoch, celsius0=20.0):
    raw = np.full((n, 4, 5), int(celsius0 * 100), dtype=np.uint16)
    timestamps = start_epoch + np.arange(n, dtype=np.float64) / 8.0
    seqs = np.arange(n, dtype=np.uint64)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=raw)
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset("sequence_numbers", data=seqs)
        f.create_group("metadata")


def _make_room(tmp_path, room_id, date, n_frames_per_cam=16, start_dt=None):
    """Build a minimal 3-camera HDF5 session (single chunk per camera) plus
    a matching labels.csv: first half Empty, second half Contact_2+Humans."""
    start_dt = start_dt or dt.datetime(2026, 6, 30, 12, 0, 0)
    room_root = tmp_path / room_id
    date_dir = room_root / date
    for cam in (0, 1, 2):
        cam_dir = date_dir / f"cam_{cam}"
        cam_dir.mkdir(parents=True)
        _write_chunk(cam_dir / "120000_seq0000000_seq0000015.h5", n=n_frames_per_cam, start_epoch=start_dt.timestamp())

    half = n_frames_per_cam // 2
    mid = start_dt + dt.timedelta(seconds=half / 8.0)
    end = start_dt + dt.timedelta(seconds=n_frames_per_cam / 8.0 + 1)
    csv_path = room_root / "labels.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        fh.write(f"{room_id},{date},{start_dt.strftime('%H:%M:%S.%f')[:-3]},{mid.strftime('%H:%M:%S.%f')[:-3]},Empty,10,0|1|2,x\n")
        fh.write(f"{room_id},{date},{mid.strftime('%H:%M:%S.%f')[:-3]},{end.strftime('%H:%M:%S.%f')[:-3]},Contact_2+Humans,2,0|1|2,x\n")
    return room_root, csv_path


class TestDiscoverHDF5Sessions:
    def test_finds_date_folders_with_cam_dirs(self, tmp_path):
        room_root, _csv = _make_room(tmp_path, "synth_room_1", "2026-06-30")
        refs = discover_hdf5_sessions(room_root, "synth_room_1")
        assert len(refs) == 1
        assert refs[0].date == "2026-06-30"
        assert refs[0].room_id == "synth_room_1"

    def test_empty_for_missing_root(self, tmp_path):
        assert discover_hdf5_sessions(tmp_path / "nonexistent", "x") == []

    def test_ignores_date_dirs_without_cam_subdirs(self, tmp_path):
        room_root = tmp_path / "synth_room_1"
        (room_root / "not_a_session").mkdir(parents=True)
        refs = discover_hdf5_sessions(room_root, "synth_room_1")
        assert refs == []


class TestHDF5Session:
    def test_scene_property(self, tmp_path):
        room_root, csv_path = _make_room(tmp_path, "synth_room_1", "2026-06-30")
        ref = discover_hdf5_sessions(room_root, "synth_room_1")[0]
        sess = HDF5Session(ref, csv_path)
        assert sess.scene == "synth_room_1/2026-06-30"
        assert sess.cameras == (0, 1, 2)

    def test_contact_examples_match_label_intervals(self, tmp_path):
        room_root, csv_path = _make_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=16)
        ref = discover_hdf5_sessions(room_root, "synth_room_1")[0]
        sess = HDF5Session(ref, csv_path)
        examples = list(sess.contact_examples())
        assert len(examples) == 16
        # First half should be Empty (no contact); second half Contact.
        first_half_contact = [e.any_contact for _frames, e in examples[:8]]
        second_half_contact = [e.any_contact for _frames, e in examples[8:]]
        assert not any(first_half_contact)
        assert all(second_half_contact)

    def test_fire_examples_all_safe_when_no_fire_class(self, tmp_path):
        room_root, csv_path = _make_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=16)
        ref = discover_hdf5_sessions(room_root, "synth_room_1")[0]
        sess = HDF5Session(ref, csv_path)
        examples = list(sess.fire_examples())
        assert len(examples) == 16 * 3  # 3 cameras
        assert all(alert.level == FireLevel.SAFE for _frame, alert in examples)
        assert all(alert.blob_features == {} for _frame, alert in examples)

    def test_human_presence_examples_match_contact_labels(self, tmp_path):
        room_root, csv_path = _make_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=16)
        ref = discover_hdf5_sessions(room_root, "synth_room_1")[0]
        sess = HDF5Session(ref, csv_path)
        examples = list(sess.human_presence_examples())
        assert len(examples) == 16 * 3
        # Both Empty and Contact_2+Humans intervals: only Empty is human-negative.
        n_positive = sum(1 for _f, present in examples if present)
        assert n_positive == 8 * 3  # second half of each camera's frames

    def test_raises_without_any_cam_dir(self, tmp_path):
        room_root = tmp_path / "synth_room_1"
        date_dir = room_root / "2026-06-30"
        date_dir.mkdir(parents=True)
        (room_root / "labels.csv").write_text(CSV_HEADER)
        ref = HDF5SessionRef(room_root=room_root, room_id="synth_room_1", date="2026-06-30", date_dir=date_dir)
        with pytest.raises(ValueError, match="no cam_N directories"):
            HDF5Session(ref, room_root / "labels.csv")


class TestMultiSourceContactDataset:
    def test_merges_multiple_hdf5_sessions(self, tmp_path):
        room_root1, csv1 = _make_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=16)
        room_root2, csv2 = _make_room(tmp_path, "synth_room_2", "2026-06-30", n_frames_per_cam=16)
        sess1 = HDF5Session(discover_hdf5_sessions(room_root1, "synth_room_1")[0], csv1)
        sess2 = HDF5Session(discover_hdf5_sessions(room_root2, "synth_room_2")[0], csv2)

        merged = MultiSourceContactDataset(waveshare_dataset=None, hdf5_sessions=[sess1, sess2])
        all_examples = list(merged)
        assert len(all_examples) == 32  # 16 each

        counts = merged.label_counts()
        assert counts["contact"] == 16  # 8 per session
        assert counts["no_contact"] == 16

        scenes = [s.scene for s, _examples in merged.by_session()]
        assert scenes == ["synth_room_1/2026-06-30", "synth_room_2/2026-06-30"]

    def test_works_with_no_hdf5_sessions(self):
        merged = MultiSourceContactDataset()
        assert list(merged) == []
        assert merged.label_counts() == {"contact": 0, "no_contact": 0}


class TestMultiSourceFireDataset:
    def test_merges_and_iterates(self, tmp_path):
        room_root, csv_path = _make_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=16)
        sess = HDF5Session(discover_hdf5_sessions(room_root, "synth_room_1")[0], csv_path)
        merged = MultiSourceFireDataset(waveshare_dataset=None, hdf5_sessions=[sess])
        examples = list(merged)
        assert len(examples) == 16 * 3
        by_session = list(merged.by_session())
        assert len(by_session) == 1
        assert by_session[0][0].scene == "synth_room_1/2026-06-30"
        assert len(by_session[0][1]) == 16 * 3
