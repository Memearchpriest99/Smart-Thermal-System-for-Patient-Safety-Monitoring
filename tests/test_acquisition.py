"""Tests for the MI48 acquisition module (hardware mocked)."""

from __future__ import annotations

import numpy as np
import pytest

from thermal_algorithms.acquisition import (
    AcquisitionError,
    MI48Camera,
    MI48CameraConfig,
    SessionRecorder,
    mi48_data_to_array,
)
from thermal_algorithms.core.types import Frame

FPA_SHAPE = (80, 62)  # (cols, rows), as reported by the MI48


# ---------------------------------------------------------------------------
# Fakes for the hardware stack
# ---------------------------------------------------------------------------

class FakePin:
    def __init__(self):
        self.states: list[bool] = []

    def on(self):
        self.states.append(True)

    def off(self):
        self.states.append(False)


class FakeDataReady:
    def __init__(self):
        self.waits = 0

    def wait_for_active(self):
        self.waits += 1


class FakeMI48:
    """Emulates senxor.mi48.MI48: read() yields 1-D °C data + header."""

    def __init__(self, frames_c: list[np.ndarray], fail_at: int | None = None):
        self.fpa_shape = FPA_SHAPE
        self._frames = frames_c   # list of (rows, cols) °C arrays
        self._i = 0
        self._fail_at = fail_at
        self.started = False
        self.stopped = False

    def start(self, stream=True, with_header=True):
        self.started = True

    def stop(self, stop_timeout=0.5):
        self.stopped = True

    def read(self):
        if self._fail_at is not None and self._i == self._fail_at:
            return None, None
        frame = self._frames[self._i % len(self._frames)]
        self._i += 1
        # MI48 streams the FPA column-major: invert data_to_frame's reshape.
        data = frame.T.reshape(-1, order="F").astype(np.float16)
        return data, {"frame_counter": self._i}


def make_camera(camera_id: int = 0, n_frames: int = 4, fail_at: int | None = None):
    rng = np.random.default_rng(camera_id)
    frames = [
        (20 + 10 * rng.random((FPA_SHAPE[1], FPA_SHAPE[0]))).astype(np.float32)
        for _ in range(n_frames)
    ]
    cam = MI48Camera(
        MI48CameraConfig(camera_id=camera_id, spi_cs_delay_s=0.0),
        _mi48=FakeMI48(frames, fail_at=fail_at),
        _cs_pin=FakePin(),
        _data_ready=FakeDataReady(),
    )
    return cam, frames


# ---------------------------------------------------------------------------
# mi48_data_to_array
# ---------------------------------------------------------------------------

class TestDataToArray:
    def test_roundtrip_matches_source_frame(self):
        src = np.arange(80 * 62, dtype=np.float32).reshape(62, 80)
        data = src.T.reshape(-1, order="F")
        out = mi48_data_to_array(data, FPA_SHAPE)
        assert out.shape == (62, 80)
        np.testing.assert_array_equal(out, src)

    def test_hflip(self):
        src = np.arange(80 * 62, dtype=np.float32).reshape(62, 80)
        data = src.T.reshape(-1, order="F")
        out = mi48_data_to_array(data, FPA_SHAPE, hflip=True)
        np.testing.assert_array_equal(out, src[:, ::-1])

    def test_output_is_float32(self):
        data = np.zeros(80 * 62, dtype=np.float16)
        assert mi48_data_to_array(data, FPA_SHAPE).dtype == np.float32


# ---------------------------------------------------------------------------
# MI48Camera
# ---------------------------------------------------------------------------

class TestMI48Camera:
    def test_read_frame_returns_frame(self):
        cam, frames = make_camera()
        cam.start()
        frame = cam.read_frame()
        assert isinstance(frame, Frame)
        assert frame.data.shape == (62, 80)
        assert frame.data.dtype == np.float32
        assert frame.camera_id == 0
        np.testing.assert_allclose(frame.data, frames[0], atol=0.05)

    def test_data_ready_gates_every_read(self):
        cam, _ = make_camera()
        cam.start()
        cam.read_frame()
        cam.read_frame()
        assert cam._data_ready.waits == 2

    def test_cs_asserted_around_read(self):
        cam, _ = make_camera()
        cam.start()
        cam.read_frame()
        assert cam._cs_pin.states == [True, False]

    def test_none_data_raises(self):
        cam, _ = make_camera(fail_at=0)
        cam.start()
        with pytest.raises(AcquisitionError):
            cam.read_frame()
        # CS must still be deasserted after the failure
        assert cam._cs_pin.states == [True, False]

    def test_header_lands_in_metadata(self):
        cam, _ = make_camera()
        cam.start()
        frame = cam.read_frame()
        assert frame.metadata["mi48_header"] == {"frame_counter": 1}

    def test_stop_forwards_to_mi48(self):
        cam, _ = make_camera()
        cam.start()
        cam.stop()
        assert cam._mi48.stopped


# ---------------------------------------------------------------------------
# MI48USBCamera (USB-C connected modules)
# ---------------------------------------------------------------------------

class TestMI48USBCamera:
    def _make(self, camera_id: int = 0, fail_at: int | None = None):
        from thermal_algorithms.acquisition import MI48USBCamera, MI48USBCameraConfig

        rng = np.random.default_rng(camera_id)
        frames = [
            (20 + 10 * rng.random((FPA_SHAPE[1], FPA_SHAPE[0]))).astype(np.float32)
            for _ in range(4)
        ]
        cam = MI48USBCamera(
            MI48USBCameraConfig(camera_id=camera_id, port="/dev/ttyFAKE0"),
            _mi48=FakeMI48(frames, fail_at=fail_at),
        )
        return cam, frames

    def test_read_frame_returns_frame(self):
        cam, frames = self._make(camera_id=2)
        cam.start()
        frame = cam.read_frame()
        assert isinstance(frame, Frame)
        assert frame.data.shape == (62, 80)
        assert frame.camera_id == 2
        np.testing.assert_allclose(frame.data, frames[0], atol=0.05)

    def test_none_data_raises(self):
        from thermal_algorithms.acquisition import AcquisitionError

        cam, _ = self._make(fail_at=0)
        cam.start()
        with pytest.raises(AcquisitionError):
            cam.read_frame()

    def test_works_with_session_recorder(self, tmp_path):
        from thermal_algorithms.acquisition import SessionRecorder

        cams = []
        for ch in range(3):
            cam, _ = self._make(camera_id=ch)
            cam.start()
            cams.append(cam)
        rec = SessionRecorder(cams, root=tmp_path, scene="usb_test",
                              write_pngs=False)
        out = rec.record(n_frames=3)
        for ch in range(3):
            assert np.load(out / f"ch{ch}_thermal.npz")["frames"].shape == (3, 62, 80)


# ---------------------------------------------------------------------------
# SessionRecorder
# ---------------------------------------------------------------------------

class TestSessionRecorder:
    def _make_recorder(self, tmp_path, n_cams=3):
        cams = []
        for ch in range(n_cams):
            cam, _ = make_camera(camera_id=ch, n_frames=8)
            cam.start()
            cams.append(cam)
        return SessionRecorder(
            cams, root=tmp_path, scene="unit_test", write_pngs=False
        )

    def test_layout_and_alignment(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        out = rec.record(n_frames=5)
        assert out == tmp_path / "unit_test" / rec.session
        for ch in range(3):
            npz = np.load(out / f"ch{ch}_thermal.npz")
            frames = npz["frames"]
            assert frames.shape == (5, 62, 80)
            assert frames.dtype == np.float32

    def test_session_name_format(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        assert len(rec.session) == 15 and rec.session[8] == "_"
        assert rec.session.replace("_", "").isdigit()

    def test_requires_exactly_one_budget(self, tmp_path):
        rec = self._make_recorder(tmp_path)
        with pytest.raises(ValueError):
            rec.record()
        with pytest.raises(ValueError):
            rec.record(duration_s=1.0, n_frames=5)

    def test_on_tick_called_per_frame(self, tmp_path):
        rec = self._make_recorder(tmp_path, n_cams=1)
        seen = []
        rec.record(n_frames=3, on_tick=lambda i, frames: seen.append((i, len(frames))))
        assert seen == [(0, 1), (1, 1), (2, 1)]

    def test_loadable_by_training_stack_after_reorg_convention(self, tmp_path):
        """The npz key and dtype match what datasets.py expects."""
        rec = self._make_recorder(tmp_path, n_cams=1)
        out = rec.record(n_frames=2)
        arr = np.load(out / "ch0_thermal.npz")["frames"]
        frame = Frame(data=arr[0].astype(np.float32), timestamp=0.0, camera_id=0)
        assert frame.data.shape == (62, 80)
