"""Tests for SessionMetadata's npz-loading behavior, including the frames
cache added to keep whole-dataset iteration practical (previously every
single `load_frame()` call re-opened and fully re-decompressed the npz
archive — negligible per-call, but O(n_frames) over a real session)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.training.datasets import DatasetIndex, _load_npz_frames_cached


def _write_npz(path: Path, n_frames: int, profile=MLX90640, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = profile.resolution
    frames = np.arange(n_frames * h * w, dtype=np.float32).reshape(n_frames, h, w) + offset
    np.savez_compressed(path, frames=frames)


@pytest.fixture(autouse=True)
def _clear_npz_cache():
    """The cache is module-global (keyed by path) — clear it around each
    test so tests using the same tmp_path-derived filenames don't leak
    stale entries into one another."""
    _load_npz_frames_cached.cache_clear()
    yield
    _load_npz_frames_cached.cache_clear()


class TestSessionFrameLoading:
    def test_load_frame_returns_correct_data(self, tmp_path):
        _write_npz(tmp_path / "empty_room" / "ch0_raw_data.npz", n_frames=5)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        session = index.find("empty_room")

        expected = np.load(tmp_path / "empty_room" / "ch0_raw_data.npz")["frames"]
        for i in range(5):
            frame = session.load_frame(0, i)
            np.testing.assert_array_equal(frame.data, expected[i])
            assert frame.timestamp == pytest.approx(i / 8.0)
            assert frame.camera_id == 0

    def test_load_frames_returns_full_stack(self, tmp_path):
        _write_npz(tmp_path / "empty_room" / "ch0_raw_data.npz", n_frames=5)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        session = index.find("empty_room")

        expected = np.load(tmp_path / "empty_room" / "ch0_raw_data.npz")["frames"]
        np.testing.assert_array_equal(session.load_frames(0), expected)

    def test_repeated_load_frame_calls_hit_cache(self, tmp_path, monkeypatch):
        """The whole point of the cache: many load_frame() calls against the
        same npz path should decompress the archive once, not once per call."""
        _write_npz(tmp_path / "empty_room" / "ch0_raw_data.npz", n_frames=20)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        session = index.find("empty_room")

        real_np_load = np.load
        call_count = {"n": 0}

        def counting_load(*args, **kwargs):
            call_count["n"] += 1
            return real_np_load(*args, **kwargs)

        monkeypatch.setattr(np, "load", counting_load)

        for i in range(20):
            session.load_frame(0, i)

        assert call_count["n"] == 1, (
            f"expected exactly 1 underlying np.load() call for 20 load_frame() "
            f"calls against the same file, got {call_count['n']}"
        )

    def test_different_channels_are_cached_independently(self, tmp_path):
        _write_npz(tmp_path / "empty_room" / "ch0_raw_data.npz", n_frames=3)
        _write_npz(tmp_path / "empty_room" / "ch1_raw_data.npz", n_frames=3, offset=1000.0)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        session = index.find("empty_room")

        f0 = session.load_frame(0, 0)
        f1 = session.load_frame(1, 0)
        assert not np.array_equal(f0.data, f1.data)  # distinct content, distinct cache entries

    def test_astype_does_not_mutate_cached_array(self, tmp_path):
        """load_frame/load_frames return float32 copies — mutating the
        result must not corrupt the cached original for later callers."""
        _write_npz(tmp_path / "empty_room" / "ch0_raw_data.npz", n_frames=3)
        index = DatasetIndex(tmp_path, sensor_profile=MLX90640, fps=8.0)
        session = index.find("empty_room")

        frame = session.load_frame(0, 0)
        frame.data[0, 0] = -999.0
        fresh = session.load_frame(0, 0)
        assert fresh.data[0, 0] != -999.0
