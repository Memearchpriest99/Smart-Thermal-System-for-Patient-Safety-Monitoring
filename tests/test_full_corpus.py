"""Tests for the full-corpus streaming used by Phase 3/4 training scripts."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
import hdf5plugin  # noqa: E402,F401  registers the Zstd filter; import after skip-guard

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import FireLevel, Frame
from thermal_algorithms.training import full_corpus
from thermal_algorithms.training.datasets import DatasetIndex

CSV_HEADER = "Room_ID,Date,Start_Time,End_Time,Event_Class,Event_Class_ID,Cameras,Timestamp\n"


def _write_chunk(path, *, n, start_epoch, celsius0=20.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.full((n, 4, 5), int(celsius0 * 100), dtype=np.uint16)
    timestamps = start_epoch + np.arange(n, dtype=np.float64) / 8.0
    seqs = np.arange(n, dtype=np.uint64)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=raw)
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset("sequence_numbers", data=seqs)
        f.create_group("metadata")


def _make_synth_room(data_root: Path, room_id: str, date: str, n_frames_per_cam: int = 20):
    """A room with one Empty-then-Contact-then-Fire day, 3 cams, 1 chunk each."""
    start_dt = dt.datetime(2026, 6, 30, 12, 0, 0)
    room_root = data_root / room_id
    date_dir = room_root / date
    for cam in (0, 1, 2):
        _write_chunk(
            date_dir / f"cam_{cam}" / "120000_seq0000000_seq0000019.h5",
            n=n_frames_per_cam, start_epoch=start_dt.timestamp(),
        )
    third = n_frames_per_cam // 3
    t0 = start_dt
    t1 = start_dt + dt.timedelta(seconds=third / 8.0)
    t2 = start_dt + dt.timedelta(seconds=2 * third / 8.0)
    t3 = start_dt + dt.timedelta(seconds=(n_frames_per_cam / 8.0 + 1))
    with (room_root / "labels.csv").open("w", encoding="utf-8") as fh:
        fh.write(CSV_HEADER)
        fh.write(f"{room_id},{date},{t0.strftime('%H:%M:%S.%f')[:-3]},{t1.strftime('%H:%M:%S.%f')[:-3]},Empty,10,0|1|2,x\n")
        fh.write(f"{room_id},{date},{t1.strftime('%H:%M:%S.%f')[:-3]},{t2.strftime('%H:%M:%S.%f')[:-3]},Contact_2+Humans,2,0|1|2,x\n")
        fh.write(f"{room_id},{date},{t2.strftime('%H:%M:%S.%f')[:-3]},{t3.strftime('%H:%M:%S.%f')[:-3]},Fire,5,0|1|2,x\n")


def _make_empty_waveshare_index(tmp_path) -> DatasetIndex:
    """A DatasetIndex over an empty root — no waveshare scenes, so tests only
    exercise the synthetic-corpus half without needing real waveshare fixtures."""
    root = tmp_path / "waveshare_empty"
    root.mkdir()
    return DatasetIndex(root, sensor_profile=MLX90640, fps=8.0)


class TestIterSynthSessions:
    def test_discovers_sessions_across_configured_rooms(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1", "synth_room_2"))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30")
        _make_synth_room(tmp_path, "synth_room_2", "2026-06-30")

        sessions = list(full_corpus.iter_synth_sessions(tmp_path))
        assert sorted(s.scene for s in sessions) == [
            "synth_room_1/2026-06-30", "synth_room_2/2026-06-30",
        ]

    def test_skips_rooms_with_no_labels_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1", "synth_room_missing"))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30")
        # synth_room_missing directory doesn't even exist.
        sessions = list(full_corpus.iter_synth_sessions(tmp_path))
        assert len(sessions) == 1


class TestStreamFireExamples:
    def test_yields_frame_alert_pairs_from_synth(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1",))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=21)
        idx = _make_empty_waveshare_index(tmp_path)

        examples = list(full_corpus.stream_fire_examples(tmp_path, idx))
        assert len(examples) == 21 * 3  # 3 cameras
        for frame, alert in examples:
            assert isinstance(frame, Frame)
            assert alert.level in (FireLevel.SAFE, FireLevel.ACTIVE_COMBUSTION)
        # Last third of the session is labeled Fire.
        n_fire = sum(1 for _f, a in examples if a.level != FireLevel.SAFE)
        assert n_fire > 0

    def test_is_a_true_generator_not_a_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1",))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30")
        idx = _make_empty_waveshare_index(tmp_path)
        import types
        gen = full_corpus.stream_fire_examples(tmp_path, idx)
        assert isinstance(gen, types.GeneratorType)


class TestStreamContactExamples:
    def test_yields_triplet_event_pairs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1",))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=21)
        idx = _make_empty_waveshare_index(tmp_path)

        examples = list(full_corpus.stream_contact_examples(tmp_path, idx))
        assert len(examples) == 21
        for frames, event in examples:
            assert len(frames) == 3
        n_contact = sum(1 for _f, e in examples if e.any_contact)
        assert n_contact > 0


class TestIterContactTrainingChunks:
    def test_chunks_respect_chunk_size(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1",))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=25)
        idx = _make_empty_waveshare_index(tmp_path)

        chunks = list(full_corpus.iter_contact_training_chunks(tmp_path, idx, chunk_size=10))
        sizes = [len(frames) for _label, frames, _events in chunks]
        assert sizes == [10, 10, 5]
        assert all(label == "synth_room_1/2026-06-30" for label, _f, _e in chunks)

    def test_does_not_mix_sessions_in_one_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1", "synth_room_2"))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=8)
        _make_synth_room(tmp_path, "synth_room_2", "2026-06-30", n_frames_per_cam=8)
        idx = _make_empty_waveshare_index(tmp_path)

        # chunk_size larger than either session -> one chunk per session, never combined.
        chunks = list(full_corpus.iter_contact_training_chunks(tmp_path, idx, chunk_size=1000))
        labels = [label for label, _f, _e in chunks]
        assert labels == ["synth_room_1/2026-06-30", "synth_room_2/2026-06-30"]

    def test_total_examples_conserved_across_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(full_corpus, "SYNTH_ROOMS", ("synth_room_1",))
        _make_synth_room(tmp_path, "synth_room_1", "2026-06-30", n_frames_per_cam=17)
        idx = _make_empty_waveshare_index(tmp_path)

        chunks = list(full_corpus.iter_contact_training_chunks(tmp_path, idx, chunk_size=6))
        total = sum(len(frames) for _label, frames, _events in chunks)
        assert total == 17
