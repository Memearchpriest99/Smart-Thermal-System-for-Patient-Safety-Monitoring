"""Tests for the Live Thermal Monitor app (apps/live_monitor).

The capture / detector / rendering / runner / calibration layers are Qt-free and
tested directly. A small set of widget tests runs under the offscreen Qt
platform and is skipped if PyQt6 is unavailable.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.checkpoints import CheckpointRegistry
from thermal_algorithms.core.types import HomographyMatrices

from apps.live_monitor import rendering
from apps.live_monitor.capture import PlaybackSource, SyntheticSource
from apps.live_monitor.detectors import BuildContext, build_registry
from apps.live_monitor.runner import PipelineRunner
from apps.live_monitor import calibration as calib
from apps.live_monitor.ui import room3d

PROFILE = WAVESHARE_26984


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #

def _read_until(src, tries=500, sleep=0.002):
    for _ in range(tries):
        f = src.read()
        if f is not None:
            return f
        time.sleep(sleep)
    return None


def test_synthetic_source_shape_and_pacing():
    src = SyntheticSource(0, PROFILE, with_fire=True, seed=1).start()
    f = _read_until(src)
    assert f is not None
    assert f.data.shape == (PROFILE.height, PROFILE.width)
    assert f.data.dtype == np.float32
    # the fire blob pushes the max well above body temperature
    assert f.data.max() > 80.0
    # immediately after a read, the next read is throttled (no new frame yet)
    assert src.read() is None
    src.stop()
    assert src.health.frames_read >= 1


def test_playback_source_roundtrip_and_loop(tmp_path):
    frames = np.stack([np.full((PROFILE.height, PROFILE.width), 20.0 + i, np.float32)
                       for i in range(4)])
    np.savez(tmp_path / "ch0_raw_data.npz", frames=frames)
    src = PlaybackSource(0, PROFILE, session_dir=tmp_path, loop=True).start()
    seen = []
    for _ in range(6):
        f = _read_until(src)
        assert f is not None
        seen.append(float(f.data[0, 0]))
    src.stop()
    assert seen[0] == pytest.approx(20.0)
    assert seen[3] == pytest.approx(23.0)
    assert seen[4] == pytest.approx(20.0)  # looped back


def test_playback_source_missing_file(tmp_path):
    src = PlaybackSource(2, PROFILE, session_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        src.start()


# --------------------------------------------------------------------------- #
# detector registry
# --------------------------------------------------------------------------- #

def test_registry_rule_based_available_trainable_not(tmp_path):
    reg = CheckpointRegistry(tmp_path / "ckpts")  # empty → no checkpoints
    ctx = BuildContext(profile=PROFILE, registry=reg)
    options = build_registry()
    for task, opts in options.items():
        rule_based = opts[0]
        ok, _ = rule_based.availability(ctx)
        assert ok, f"{task} rule-based option should always be available"
        assert rule_based.build(ctx) is not None
        # every trainable option is unavailable with an empty registry
        for opt in opts[1:]:
            avail, reason = opt.availability(ctx)
            assert not avail
            assert reason


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def _runner(tmp_path, **kw):
    reg = CheckpointRegistry(tmp_path / "ckpts")
    srcs = [SyntheticSource(i, PROFILE, with_fire=(i == 0), seed=i) for i in range(3)]
    return PipelineRunner(srcs, PROFILE, reg, **kw)


def test_runner_defaults_and_process(tmp_path):
    runner = _runner(tmp_path)
    assert runner.selection == {
        "fire": "otsu_fire_detector",
        "person": "adaptive_threshold_detector",
        "touch": "geometric_contact_detector",
    }
    runner.start()
    got = None
    for _ in range(500):
        got = runner.poll()
        if got is not None:
            break
        time.sleep(0.002)
    runner.stop()
    assert got is not None
    assert len(got.result.detections) == 3
    assert "pipeline" in got.timings_ms


def test_runner_thresholds_applied_to_geometric(tmp_path):
    runner = _runner(tmp_path)
    runner.set_thresholds(delta_m=0.73, epsilon_m=0.21)
    contact = runner.pipeline._contact_detector
    assert contact._delta_m == pytest.approx(0.73)
    assert contact._epsilon_m == pytest.approx(0.21)
    assert runner._delta_m == pytest.approx(0.73)


def test_runner_restricted_mode(tmp_path):
    runner = _runner(tmp_path, restricted=False)
    assert not runner.pipeline.restricted
    runner.set_restricted(True)
    assert runner.pipeline.restricted


def test_runner_is_geometric_touch(tmp_path):
    runner = _runner(tmp_path)
    assert runner.is_geometric_touch()


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

def test_colorize_shape_dtype_range():
    frame = np.linspace(20, 60, PROFILE.height * PROFILE.width, dtype=np.float32)
    frame = frame.reshape(PROFILE.height, PROFILE.width)
    rgb = rendering.colorize(frame)
    assert rgb.shape == (PROFILE.height, PROFILE.width, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb.min()) >= 0 and int(rgb.max()) <= 255
    assert rendering.INFERNO_LUT.shape == (256, 3)


def test_colorize_flat_frame_is_safe():
    flat = np.full((10, 12), 25.0, np.float32)
    rgb = rendering.colorize(flat)
    assert rgb.shape == (10, 12, 3)


# --------------------------------------------------------------------------- #
# calibration + 3-D room
# --------------------------------------------------------------------------- #

def _exact_corr(scale=10.0):
    """Pixel↔floor pairs that follow an exact affine map, per camera."""
    corners = calib.rectangle_world_corners((2.0, 1.0))
    rows = []
    for (X, Y) in corners:
        rows.append((X * scale + 5.0, Y * scale + 5.0, X, Y))
    return rows


def test_solve_homographies_exact_and_roundtrip(tmp_path):
    inp = calib.CalibrationInput(
        room_lwh=(4.0, 4.0, 2.8),
        rect_wh=(2.0, 1.0),
        correspondences={0: _exact_corr(10), 1: _exact_corr(11), 2: _exact_corr(9)},
        camera_poses={c: (0.2, 2.5, 2.6, 0.0, 30.0, 45.0, 34.0) for c in range(3)},
        ref_camera=0,
    )
    H, info = calib.solve_homographies(inp)
    for c in range(3):
        assert info[c]["median_residual"] < 1e-6  # exact affine → near-zero residual

    out = calib.save_calibration(tmp_path / "homography_calibration.npz", inp, H)
    assert out.exists()
    H2, extras = calib.load_calibration(out)
    assert np.allclose(H.h1, H2.h1)
    assert tuple(extras["rect_wh"]) == (2.0, 1.0)
    assert tuple(extras["room_lwh"]) == (4.0, 4.0, 2.8)
    assert extras["pairs_cam0"].shape == (4, 4)


def test_solve_requires_four_points():
    inp = calib.CalibrationInput(
        room_lwh=(3, 3, 2.5), rect_wh=(1.0, 1.0),
        correspondences={0: [(0, 0, 0, 0), (1, 1, 1, 0)]},  # only 2
        camera_poses={}, ref_camera=0,
    )
    with pytest.raises(ValueError):
        calib.solve_homographies(inp)


def test_room_clamp_caps_at_five():
    assert room3d.clamp_room(9, -1, 100) == (5.0, 0.1, 5.0)
    assert room3d.ROOM_MAX_M == 5.0


def test_box_edges_and_camera_pose():
    edges = room3d.box_edges(4, 3, 2.5)
    assert len(edges) == 12
    assert all(e.shape == (2, 3) for e in edges)
    cam = room3d.CameraPose(0, 0.2, 2.5, 2.6, yaw_deg=0.0, tilt_deg=90.0)
    fwd = cam.forward()
    assert fwd[2] == pytest.approx(-1.0, abs=1e-6)  # straight down
    assert cam.frustum_corners().shape == (4, 3)


def test_foot_point_projection_consistency():
    """Identity homography → world coords equal pixel coords (bird's-eye fallback)."""
    from thermal_algorithms.contact_detection.multi_view.homography import project_foot_point
    H = np.eye(3)
    assert project_foot_point((12.0, 34.0), H) == (12.0, 34.0)


# --------------------------------------------------------------------------- #
# Qt widgets (offscreen; skipped without PyQt6)
# --------------------------------------------------------------------------- #

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pyqt = pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    app = pyqt.QApplication.instance() or pyqt.QApplication([])
    yield app


def test_to_qimage(qapp):
    rgb = np.zeros((5, 7, 3), np.uint8)
    img = rendering.to_qimage(rgb)
    assert img.width() == 7 and img.height() == 5


def test_camera_view_set_frame(qapp):
    from apps.live_monitor.ui.camera_view import CameraView
    from thermal_algorithms.core.types import Detection
    view = CameraView(0)
    view.resize(160, 120)
    rgb = rendering.colorize(np.random.rand(PROFILE.height, PROFILE.width).astype(np.float32))
    view.set_frame(rgb, temp=np.zeros((PROFILE.height, PROFILE.width), np.float32),
                   person=[Detection(bbox=(5, 5, 10, 20), score=0.9, class_id=0)],
                   fire=[Detection(bbox=(1, 1, 3, 3), score=0.8, class_id=0)])
    view.grab()  # exercises paintEvent


def test_birdseye_widget_update(qapp):
    from apps.live_monitor.ui.birdseye_widget import BirdsEyeWidget
    from thermal_algorithms.core.types import ActorPosition, ContactEvent, Detection
    w = BirdsEyeWidget()
    w.resize(300, 300)
    H = HomographyMatrices(h1=np.eye(3), h2=np.eye(3), h3=np.eye(3))
    w.set_geometry(H, corners=np.array([(0, 0), (2, 0), (2, 1), (0, 1)]), unit="m", delta_m=0.5)
    ev = ContactEvent(
        actors=(ActorPosition((0.5, 0.5), 0), ActorPosition((0.7, 0.5), 1)),
        pairs_in_contact=((0, 1),), timestamp=0.0, confidence=1.0)
    dets = [[Detection(bbox=(10, 10, 4, 8))], [], []]
    w.update_frame(dets, ev)
    w.grab()


def test_calibration_dialog_build_input(qapp):
    from apps.live_monitor.ui.calibration_dialog import CalibrationDialog
    dlg = CalibrationDialog()
    inp = dlg.build_input()
    assert len(inp.room_lwh) == 3
    assert all(0 < d <= room3d.ROOM_MAX_M for d in inp.room_lwh)
    assert set(inp.correspondences.keys()) == {0, 1, 2}
