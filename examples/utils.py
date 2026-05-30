"""Shared utilities for all demo scripts.

Provides:
  - load_frames()               — load an NPZ session channel into Frame objects
  - find_session()              — locate the first matching session directory
  - make_synthetic_homographies() — build a calibration-free homography for demos
  - make_synthetic_triplet()    — generate a 3-camera frame triplet with
                                  persons at given normalised positions
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thermal_algorithms.contact_detection.multi_view.homography import (
    solve_homography_from_markers,
)
from thermal_algorithms.core.sensor_profile import MLX90640, SensorProfile
from thermal_algorithms.core.types import Detection, Frame, HomographyMatrices


# ---------------------------------------------------------------------------
# NPZ / session loading
# ---------------------------------------------------------------------------

def load_frames(
    npz_path: Path,
    camera_id: int,
    fps: float = 8.0,
) -> list[Frame]:
    """Load all frames from a ch{N}_raw_data.npz into a list of Frame objects."""
    arr = np.load(npz_path)["frames"].astype(np.float32)
    return [
        Frame(data=arr[i], timestamp=i / fps, camera_id=camera_id)
        for i in range(arr.shape[0])
    ]


def find_session(
    candidates: list[Path],
    require_channels: int = 1,
) -> Path | None:
    """Return the first directory that has data for all required channels.

    Handles both flat layouts (channel npz files directly in ``candidate``)
    and timestamped sub-folder layouts (``candidate/20260405_195559/ch0_*.npz``).
    """
    for candidate in candidates:
        if not candidate.exists():
            continue
        for root in [candidate] + sorted(candidate.iterdir()
                                         if candidate.is_dir() else []):
            if not root.is_dir():
                continue
            if all((root / f"ch{ch}_raw_data.npz").is_file()
                   for ch in range(require_channels)):
                return root
    return None


# ---------------------------------------------------------------------------
# Synthetic homography builder (no physical calibration required)
# ---------------------------------------------------------------------------

def make_synthetic_homographies(
    profile: SensorProfile = MLX90640,
    room_w_m: float = 4.0,
    room_h_m: float = 3.0,
) -> HomographyMatrices:
    """Build three camera homographies for a synthetic room.

    Cameras are placed at three distinct vantage points of a
    ``room_w_m × room_h_m`` floor plan:

      Camera 0 — top-left corner,   maps pixels to the full floor
      Camera 1 — top-right corner,  horizontally mirrored
      Camera 2 — bottom-centre,     rotated 90 °

    Replace the return value with the output of
    ``detector.calibrate_homography(marker_correspondences)`` for any
    real deployment using Hot-Point Calibration markers.
    """
    W, H = float(profile.width), float(profile.height)
    RW, RH = room_w_m, room_h_m

    cam0 = [(0, [
        ((0.0,  0.0),  (0.0,  0.0)),
        ((W,    0.0),  (RW,   0.0)),
        ((0.0,  H),    (0.0,  RH)),
        ((W,    H),    (RW,   RH)),
        ((W/2,  H/2),  (RW/2, RH/2)),
    ])]
    cam1 = [(1, [
        ((0.0,  0.0),  (RW,   0.0)),
        ((W,    0.0),  (0.0,  0.0)),
        ((0.0,  H),    (RW,   RH)),
        ((W,    H),    (0.0,  RH)),
        ((W/2,  H/2),  (RW/2, RH/2)),
    ])]
    cam2 = [(2, [
        ((0.0,  0.0),  (0.0,  RH)),
        ((W,    0.0),  (RW,   RH)),
        ((0.0,  H),    (0.0,  0.0)),
        ((W,    H),    (RW,   0.0)),
        ((W/2,  H/2),  (RW/2, RH/2)),
    ])]

    return solve_homography_from_markers(cam0 + cam1 + cam2)


# ---------------------------------------------------------------------------
# Synthetic frame generation
# ---------------------------------------------------------------------------

def make_synthetic_triplet(
    person_positions_norm: list[tuple[float, float]],
    profile: SensorProfile = MLX90640,
    timestamp: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[Frame, Frame, Frame], list[list[Detection]]]:
    """Generate a 3-camera frame triplet with person blobs at given positions.

    Args:
        person_positions_norm: ``[(cx_norm, cy_norm)]`` in the world/floor
            frame [0, 1].  Each person gets a mild per-camera perspective
            perturbation to simulate real multi-view noise.
        profile: Sensor profile (sets frame resolution).
        timestamp: Timestamp for all three frames (camera offsets added).
        rng: Optional RNG for reproducibility.

    Returns:
        ``(frames_triplet, detections_per_camera)`` — frames as a 3-tuple,
        detections as a list of three lists (one per camera).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    W, H = profile.width, profile.height
    frames: list[Frame] = []
    dets_per_cam: list[list[Detection]] = []

    for cam_id in range(3):
        data = (rng.standard_normal((H, W)) * 0.4 + 25.0).astype(np.float32)
        cam_dets: list[Detection] = []

        for cx_n, cy_n in person_positions_norm:
            perturb = rng.standard_normal(2) * 0.03
            cx = float(np.clip(cx_n + perturb[0], 0.1, 0.9))
            cy = float(np.clip(cy_n + perturb[1], 0.1, 0.9))

            bw, bh = max(2, int(W * 0.12)), max(2, int(H * 0.35))
            bx = max(0, int(cx * W - bw // 2))
            by = max(0, int(cy * H - bh // 2))
            bw = min(bw, W - bx)
            bh = min(bh, H - by)

            data[by:by + bh, bx:bx + bw] = float(rng.uniform(34.0, 37.0))
            cam_dets.append(Detection(
                bbox=(float(bx), float(by), float(bw), float(bh)),
                score=1.0, class_id=1, camera_id=cam_id,
            ))

        frames.append(Frame(data=data, timestamp=timestamp + cam_id * 0.02,
                            camera_id=cam_id))
        dets_per_cam.append(cam_dets)

    return tuple(frames), dets_per_cam  # type: ignore[return-value]
