"""Tests for the HDF5 chunk reader used by synth_room_*/room-1 sessions.

Fixtures write tiny synthetic .h5 chunks (not the real project data, which
lives outside the repo under data/) so these tests are hermetic and fast.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
import hdf5plugin  # noqa: E402,F401  registers the Zstd filter; import after skip-guard

from thermal_algorithms.training.hdf5_source import (
    DEFAULT_FALLBACK_FPS,
    HDF5CameraSession,
    _chunk_wallclock_start,
    _timestamps_look_valid,
    infer_fps_from_chunk_spacing,
    list_chunks,
    read_chunk,
)

_PLAUSIBLE_EPOCH = 1_700_000_000.0  # 2023-11-14, comfortably inside the valid range
_BOGUS_LARGE_EPOCH = 10_837_889_550.0  # the real anomaly found in room-1's last chunk


def _write_chunk(path, *, n=10, start_seq=0, start_epoch=None, celsius0=20.0, broken=False):
    """Write a minimal synthetic-style chunk: frames/timestamps/sequence_numbers."""
    raw = np.full((n, 4, 5), int(celsius0 * 100), dtype=np.uint16)
    if broken:
        timestamps = np.zeros(n, dtype=np.float64)
        seqs = np.zeros(n, dtype=np.uint64)
    else:
        assert start_epoch is not None
        timestamps = start_epoch + np.arange(n, dtype=np.float64) / 24.0
        seqs = np.arange(start_seq, start_seq + n, dtype=np.uint64)
    with h5py.File(path, "w") as f:
        f.create_dataset("frames", data=raw)
        f.create_dataset("timestamps", data=timestamps)
        f.create_dataset("sequence_numbers", data=seqs)
        meta = f.create_group("metadata")
        meta.attrs["fps"] = 24.0
        meta.attrs["synthetic"] = True


class TestListChunks:
    def test_sorts_by_leading_hhmmss(self, tmp_path):
        for name in ("120000_seq0000000_seq0000009.h5", "113000_seq0000000_seq0000009.h5"):
            _write_chunk(tmp_path / name, start_epoch=_PLAUSIBLE_EPOCH)
        chunks = list_chunks(tmp_path)
        assert [p.name for p in chunks] == [
            "113000_seq0000000_seq0000009.h5",
            "120000_seq0000000_seq0000009.h5",
        ]

    def test_rejects_unexpected_filename(self, tmp_path):
        (tmp_path / "not_a_chunk.h5").touch()
        with pytest.raises(ValueError, match="Unexpected h5 chunk filename"):
            list_chunks(tmp_path)


class TestChunkWallclockStart:
    def test_parses_hhmmss_and_date(self):
        start = _chunk_wallclock_start(
            Path("121624_seq0000000_seq0009866.h5"), "2026-06-30",
        )
        assert start == dt.datetime(2026, 6, 30, 12, 16, 24)


class TestInferFpsFromChunkSpacing:
    def test_infers_12fps_from_10min_gaps_7200_frames(self):
        # Mirrors the real room-1 pattern: chunks 600s apart, 7200 frames each
        # => 12 fps, not the synthetic data's 24 fps.
        paths = [
            Path("121624_seq0000000_seq0009866.h5"),
            Path("122624_seq0009868_seq0019809.h5"),
            Path("123624_seq0019810_seq0029749.h5"),
        ]
        lengths = [7200, 7200, 7200]
        fps = infer_fps_from_chunk_spacing(paths, lengths, "2026-06-30")
        assert fps == pytest.approx(12.0)

    def test_robust_to_one_irregular_trailing_chunk(self):
        # A short/partial final chunk (common — session just ends there)
        # shouldn't drag the median off the true rate.
        paths = [
            Path("121624_seq0000000_seq0009866.h5"),
            Path("122624_seq0009868_seq0019809.h5"),
            Path("123624_seq0019810_seq0029749.h5"),
            Path("124624_seq0029750_seq0039688.h5"),
            Path("125050_seq0039689_seq0041000.h5"),  # short trailing chunk
        ]
        lengths = [7200, 7200, 7200, 7200, 1311]
        fps = infer_fps_from_chunk_spacing(paths, lengths, "2026-06-30")
        assert fps == pytest.approx(12.0)

    def test_none_with_fewer_than_two_chunks(self):
        assert infer_fps_from_chunk_spacing(
            [Path("121624_seq0000000_seq0009866.h5")], [7200], "2026-06-30"
        ) is None


class TestTimestampsLookValid:
    def test_all_zero_is_invalid(self):
        assert not _timestamps_look_valid(np.zeros(10))

    def test_bogus_large_value_is_invalid(self):
        # The real room-1 anomaly: non-zero, internally consistent, but not
        # actually Unix-epoch seconds (implies a year-2319 date).
        assert not _timestamps_look_valid(
            _BOGUS_LARGE_EPOCH + np.arange(10, dtype=np.float64)
        )

    def test_plausible_epoch_is_valid(self):
        assert _timestamps_look_valid(_PLAUSIBLE_EPOCH + np.arange(10, dtype=np.float64))

    def test_empty_is_valid(self):
        assert _timestamps_look_valid(np.array([]))


class TestReadChunk:
    def test_scale_conversion_to_celsius(self, tmp_path):
        path = tmp_path / "070000_seq0000000_seq0000009.h5"
        _write_chunk(path, n=5, start_epoch=_PLAUSIBLE_EPOCH, celsius0=29.53)
        frames, timestamps, meta, used_fallback = read_chunk(path)
        assert not used_fallback
        assert frames.shape == (5, 4, 5)
        assert np.allclose(frames, 29.53, atol=1e-2)
        assert meta.get("synthetic") == True  # noqa: E712 (h5py attr round-trips as np.bool_)

    def test_clean_timestamps_pass_through(self, tmp_path):
        path = tmp_path / "070000_seq0000000_seq0000009.h5"
        _write_chunk(path, n=5, start_epoch=_PLAUSIBLE_EPOCH)
        _frames, timestamps, _meta, used_fallback = read_chunk(path)
        assert not used_fallback
        assert np.allclose(np.diff(timestamps), 1.0 / 24.0)

    def test_broken_timestamps_raise_without_session_date(self, tmp_path):
        path = tmp_path / "070000_seq0000000_seq0000009.h5"
        _write_chunk(path, n=5, broken=True)
        with pytest.raises(ValueError, match="session_date"):
            read_chunk(path)

    def test_broken_timestamps_fall_back_to_filename(self, tmp_path):
        path = tmp_path / "121624_seq0000000_seq0000009.h5"
        _write_chunk(path, n=5, broken=True)
        frames, timestamps, _meta, used_fallback = read_chunk(
            path, session_date="2026-06-30"
        )
        assert used_fallback
        expected_start = dt.datetime(2026, 6, 30, 12, 16, 24).timestamp()
        assert timestamps[0] == pytest.approx(expected_start)
        assert np.allclose(np.diff(timestamps), 1.0 / DEFAULT_FALLBACK_FPS)


class TestHDF5CameraSession:
    def test_indexes_chunks_without_loading_pixels(self, tmp_path):
        _write_chunk(
            tmp_path / "070000_seq0000000_seq0000004.h5",
            n=5, start_seq=0, start_epoch=_PLAUSIBLE_EPOCH, celsius0=20.0,
        )
        _write_chunk(
            tmp_path / "070500_seq0000005_seq0000009.h5",
            n=5, start_seq=5, start_epoch=_PLAUSIBLE_EPOCH + 300.0, celsius0=21.0,
        )
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        assert sess.n_frames == 10
        assert sess.n_chunks == 2
        assert not sess.used_fallback
        # Nothing decoded yet — cache should be empty until a frame is requested.
        assert len(sess._cache) == 0

    def test_load_frames_concatenates_in_order(self, tmp_path):
        _write_chunk(
            tmp_path / "070000_seq0000000_seq0000004.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH, celsius0=20.0,
        )
        _write_chunk(
            tmp_path / "070500_seq0000005_seq0000009.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH + 300.0, celsius0=21.0,
        )
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        frames = sess.load_frames()
        assert np.allclose(frames[:5], 20.0)
        assert np.allclose(frames[5:], 21.0)
        ts = sess.load_timestamps()
        assert ts[5] > ts[4]  # still increasing across the chunk boundary

    def test_load_frame_matches_bulk_arrays(self, tmp_path):
        _write_chunk(tmp_path / "070000_seq0000000_seq0000004.h5", n=5, start_epoch=_PLAUSIBLE_EPOCH)
        sess = HDF5CameraSession(tmp_path, cam_id=2)
        frame = sess.load_frame(3)
        assert frame.camera_id == 2
        assert np.array_equal(frame.data, sess.load_frames()[3])
        assert frame.timestamp == sess.load_timestamps()[3]

    def test_load_frame_out_of_range_raises(self, tmp_path):
        _write_chunk(tmp_path / "070000_seq0000000_seq0000004.h5", n=5, start_epoch=_PLAUSIBLE_EPOCH)
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        with pytest.raises(IndexError):
            sess.load_frame(5)

    def test_iter_frames_matches_load_frame(self, tmp_path):
        _write_chunk(
            tmp_path / "070000_seq0000000_seq0000004.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH, celsius0=20.0,
        )
        _write_chunk(
            tmp_path / "070500_seq0000005_seq0000009.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH + 300.0, celsius0=21.0,
        )
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        collected = list(sess.iter_frames())
        assert len(collected) == 10
        for i, frame in enumerate(collected):
            expected = sess.load_frame(i)
            assert np.array_equal(frame.data, expected.data)
            assert frame.timestamp == expected.timestamp

    def test_iter_frames_respects_start_stop(self, tmp_path):
        _write_chunk(
            tmp_path / "070000_seq0000000_seq0000004.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH, celsius0=20.0,
        )
        _write_chunk(
            tmp_path / "070500_seq0000005_seq0000009.h5",
            n=5, start_epoch=_PLAUSIBLE_EPOCH + 300.0, celsius0=21.0,
        )
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        collected = list(sess.iter_frames(start=3, stop=7))
        assert len(collected) == 4
        assert np.allclose(collected[0].data, 20.0)
        assert np.allclose(collected[-1].data, 21.0)

    def test_cache_evicts_beyond_configured_size(self, tmp_path):
        for i, hhmmss in enumerate(["070000", "070500", "071000"]):
            _write_chunk(
                tmp_path / f"{hhmmss}_seq{i * 5:07d}_seq{i * 5 + 4:07d}.h5",
                n=5, start_epoch=_PLAUSIBLE_EPOCH + i * 300.0,
            )
        sess = HDF5CameraSession(tmp_path, cam_id=0, cache_chunks=1)
        sess.load_frame(0)   # decodes chunk 0
        sess.load_frame(5)   # decodes chunk 1, evicts chunk 0
        assert 0 not in sess._cache
        assert 1 in sess._cache

    def test_clear_cache_empties_it(self, tmp_path):
        _write_chunk(tmp_path / "070000_seq0000000_seq0000004.h5", n=5, start_epoch=_PLAUSIBLE_EPOCH)
        sess = HDF5CameraSession(tmp_path, cam_id=0)
        sess.load_frame(0)
        assert len(sess._cache) == 1
        sess.clear_cache()
        assert len(sess._cache) == 0

    def test_broken_session_infers_fps_and_stays_monotonic(self, tmp_path):
        # Regression test for the real bug: assuming 24fps (the synthetic
        # rate) on a room-1-shaped session (chunks 600s apart, 7200 frames
        # each => true rate 12fps) made the reconstructed timeline go
        # backwards at every chunk boundary. Auto-inferred fps must not.
        chunk_names = [
            "121624_seq0000000_seq0009866.h5",
            "122624_seq0009868_seq0019809.h5",
            "123624_seq0019810_seq0029749.h5",
        ]
        for name in chunk_names:
            _write_chunk(tmp_path / name, n=7200, broken=True)
        sess = HDF5CameraSession(tmp_path, cam_id=0, session_date="2026-06-30")
        assert sess.used_fallback
        assert sess.fallback_fps == pytest.approx(12.0)

        prev_ts = None
        for frame in sess.iter_frames():
            if prev_ts is not None:
                assert frame.timestamp > prev_ts
            prev_ts = frame.timestamp

    def test_chains_past_a_trailing_chunk_with_inconsistent_filename(self, tmp_path):
        # Regression test for the real room-1 anomaly: a trailing chunk whose
        # own filename-declared start is EARLIER than where the regular
        # 10-min/7200-frame pattern would naturally continue to. Anchoring it
        # from its own filename (instead of chaining from the previous
        # chunk's end) makes the timeline jump backwards.
        _write_chunk(tmp_path / "121624_seq0000000_seq0009866.h5", n=7200, broken=True)
        _write_chunk(tmp_path / "122624_seq0009868_seq0019809.h5", n=7200, broken=True)
        _write_chunk(tmp_path / "123624_seq0019810_seq0029749.h5", n=7200, broken=True)
        # Natural continuation would be 12:46:24; declare an inconsistent,
        # earlier time instead, mirroring the real anomaly. With 3 regular
        # gaps (12) preceding it, the median fps stays robustly 12 despite
        # this one bad gap (33.3), same as the real 18-good/1-bad session.
        _write_chunk(tmp_path / "124000_seq0029750_seq0031000.h5", n=1200, broken=True)

        sess = HDF5CameraSession(tmp_path, cam_id=0, session_date="2026-06-30")
        assert sess.fallback_fps == pytest.approx(12.0)

        prev_ts = None
        for frame in sess.iter_frames():
            if prev_ts is not None:
                assert frame.timestamp > prev_ts, "timeline must not go backwards"
            prev_ts = frame.timestamp

        # The trailing chunk should start exactly where the previous chunk
        # ends, not at its own (inconsistent) filename time of 12:40:00.
        last_chunk_first_frame = sess.load_frame(7200 * 3)
        expected = dt.datetime(2026, 6, 30, 12, 46, 24).timestamp()
        assert last_chunk_first_frame.timestamp == pytest.approx(expected)

    def test_explicit_fallback_fps_overrides_inference(self, tmp_path):
        chunk_names = [
            "121624_seq0000000_seq0009866.h5",
            "122624_seq0009868_seq0019809.h5",
        ]
        for name in chunk_names:
            _write_chunk(tmp_path / name, n=7200, broken=True)
        sess = HDF5CameraSession(
            tmp_path, cam_id=0, session_date="2026-06-30", fallback_fps=24.0
        )
        assert sess.fallback_fps == 24.0

    def test_used_fallback_true_if_any_chunk_broken(self, tmp_path):
        _write_chunk(
            tmp_path / "121624_seq0000000_seq0000004.h5", n=5, start_epoch=_PLAUSIBLE_EPOCH,
        )
        _write_chunk(
            tmp_path / "122124_seq0000005_seq0000009.h5", n=5, broken=True,
        )
        sess = HDF5CameraSession(tmp_path, cam_id=0, session_date="2026-06-30")
        assert sess.used_fallback

    def test_construction_raises_without_session_date_if_any_chunk_broken(self, tmp_path):
        _write_chunk(
            tmp_path / "121624_seq0000000_seq0000004.h5", n=5, broken=True,
        )
        with pytest.raises(ValueError, match="session_date"):
            HDF5CameraSession(tmp_path, cam_id=0)

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No .h5 chunks"):
            HDF5CameraSession(tmp_path, cam_id=0)
