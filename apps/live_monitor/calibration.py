"""Room/homography calibration math + persistence (Qt-free, testable).

Produces a ``homography_calibration.npz`` whose ``h1/h2/h3``,
``world_positions``, ``rect_wh``, ``ref_camera`` and ``pairs_camN`` keys match
the format ``image_annotator`` writes (so the contact stack and
``scripts/visualize_birdseye.py`` load it unchanged), plus extra keys describing
the room box and camera poses for the 3-D model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from thermal_algorithms.core.types import HomographyMatrices
from thermal_algorithms.contact_detection.multi_view.homography import (
    project_foot_point,
    solve_homography_from_markers,
)


# one (u, v, X_w, Y_w) correspondence
Correspondence = tuple[float, float, float, float]


@dataclass
class CalibrationInput:
    """Everything the calibration dialog collects."""

    room_lwh: tuple[float, float, float]
    rect_wh: tuple[float, float]
    # per-camera list of (u, v, X_w, Y_w)
    correspondences: dict[int, list[Correspondence]]
    # per-camera pose: (x, y, z, yaw_deg, tilt_deg, fov_h, fov_v)
    camera_poses: dict[int, tuple[float, float, float, float, float, float, float]]
    ref_camera: int = 0


def rectangle_world_corners(rect_wh: tuple[float, float]) -> list[tuple[float, float]]:
    """Canonical floor coords for the 4 rectangle corners: TL,TR,BR,BL."""
    w, h = rect_wh
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def solve_homographies(inp: CalibrationInput) -> tuple[HomographyMatrices, dict[int, dict]]:
    """Solve per-camera homographies from the entered correspondences.

    Returns ``(HomographyMatrices, info)`` where ``info[cam]`` reports
    ``n_pairs`` and the median reprojection residual (in floor units).
    Cameras with < 4 correspondences are skipped (identity fallback).
    """
    marker_correspondences = []
    for cam, rows in sorted(inp.correspondences.items()):
        if len(rows) < 4:
            continue
        pairs = [((u, v), (X, Y)) for (u, v, X, Y) in rows]
        marker_correspondences.append((cam, pairs))
    if not marker_correspondences:
        raise ValueError("Need at least one camera with >= 4 correspondences.")

    H = solve_homography_from_markers(marker_correspondences)

    info: dict[int, dict] = {}
    for cam, rows in inp.correspondences.items():
        if len(rows) < 4:
            info[cam] = {"n_pairs": len(rows), "median_residual": None}
            continue
        residuals = []
        for (u, v, X, Y) in rows:
            px, py = project_foot_point((u, v), H[cam])
            residuals.append(float(np.hypot(px - X, py - Y)))
        info[cam] = {
            "n_pairs": len(rows),
            "median_residual": float(np.median(residuals)) if residuals else None,
        }
    return H, info


def save_calibration(path: str | Path, inp: CalibrationInput, H: HomographyMatrices) -> Path:
    """Write the npz in annotator-compatible format plus room/pose extras."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    corners = rectangle_world_corners(inp.rect_wh)
    world_positions = np.array(corners, dtype=np.float64)

    kwargs: dict[str, np.ndarray] = {
        "h1": np.asarray(H.h1, dtype=np.float64),
        "h2": np.asarray(H.h2, dtype=np.float64),
        "h3": np.asarray(H.h3, dtype=np.float64),
        "world_positions": world_positions,
        "rect_wh": np.array(inp.rect_wh, dtype=np.float64),
        "ref_camera": np.int64(inp.ref_camera),
        # extra keys (ignored by the annotator/contact loaders, used by our 3-D view)
        "room_lwh": np.array(inp.room_lwh, dtype=np.float64),
    }
    for cam in range(3):
        rows = inp.correspondences.get(cam, [])
        kwargs[f"pairs_cam{cam}"] = (
            np.array(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64)
        )
        pose = inp.camera_poses.get(cam)
        if pose is not None:
            kwargs[f"pose_cam{cam}"] = np.array(pose, dtype=np.float64)

    np.savez(path, **kwargs)
    return path


def load_calibration(path: str | Path) -> tuple[HomographyMatrices, dict]:
    """Load an npz written by us or by the annotator. Returns (H, extras)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        H = HomographyMatrices(
            h1=np.asarray(npz["h1"], dtype=np.float64),
            h2=np.asarray(npz["h2"], dtype=np.float64),
            h3=np.asarray(npz["h3"], dtype=np.float64),
        )
        extras = {k: npz[k] for k in npz.files if k not in ("h1", "h2", "h3")}
    return H, extras
