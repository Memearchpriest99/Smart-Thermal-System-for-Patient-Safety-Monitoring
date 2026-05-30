"""Hot-Point Calibration — interactive homography calibration for 3-camera setup.

Overview
--------
Physical contact detection (§ 4.4.3.1) requires a 3×3 homography matrix for
each camera that maps pixel coordinates to real-world floor coordinates (metres).
This script guides you through the "Hot-Point Calibration" procedure:

    1.  Place N ≥ 4 heated markers at known floor positions.
    2.  Record a short clip with all 3 cameras (or take single frames).
    3.  Run this script — it shows each frame and lets you click each marker.
    4.  The script computes H1, H2, H3 via DLT + SVD, reports reprojection
        error, and saves the result.

Heated markers
--------------
Standard checkerboard calibration fails in thermal (paper has uniform
emissivity).  Use objects with a clear, distinct thermal signature:

  • Small sealed containers of hot water (≈ 45–50 °C)
  • Chemical hand-warmers
  • Metal washers heated with a soldering iron (cool to safe temp first)

Place them flat on the floor at positions you have measured with a tape
measure.  Record the (X, Y) floor position of each marker in metres,
where (0, 0) is a convenient corner of the room.

Typical accuracy
----------------
With 4–6 well-spaced markers and accurate floor measurements you can
expect reprojection errors of 3–8 cm for MLX90640 sensors (limited by
the 32 × 24 resolution).  With the Waveshare (80 × 62) errors are ≤ 2 cm.

Running
-------
    python examples/calibrate_homography.py

The script reads frames from the NPZ recording you specify in CONFIGURATION
below.  After you click all markers in all three camera views a validation
plot is shown and the calibration is saved to:

    outputs/homography_calibration.npz   (numpy arrays — easy to inspect)
    outputs/homography_calibration.thalg (checkpoint for GeometricContactDetector)

Loading the saved calibration
------------------------------
    import numpy as np
    from thermal_algorithms.core.types import HomographyMatrices
    from thermal_algorithms.contact_detection.geometric import GeometricContactDetector

    cal = np.load("outputs/homography_calibration.npz")
    H = HomographyMatrices(h1=cal["h1"], h2=cal["h2"], h3=cal["h3"])
    detector = GeometricContactDetector(homography=H, epsilon_m=0.4, delta_m=0.5).fit([])
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Make sure the project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
from thermal_algorithms.contact_detection.multi_view.homography import (
    solve_homography_from_markers,
    project_foot_point,
)
from thermal_algorithms.core.types import HomographyMatrices
from thermal_algorithms.core.sensor_profile import MLX90640


# ===========================================================================
# CONFIGURATION — edit this section before running
# ===========================================================================

# Path to the recording directory that contains ch0_raw_data.npz etc.
# Leave None to use a synthetic demo (no real data needed).
RECORDING_DIR: Path | None = None
# RECORDING_DIR = Path("/sessions/magical-youthful-euler/mnt/dataset/calibration/20260405_120000")

# Frame index to use for calibration (a frame where all markers are visible).
FRAME_INDEX: int = 0

# Sensor profile of the cameras used for this recording.
SENSOR_PROFILE = MLX90640

# Floor positions of the heated markers, in metres from room corner (0, 0).
# Order matters — you will click them in this exact order in each camera view.
# Use at least 4 markers, spread across the room (avoid collinear arrangements).
#
# Example: 4 markers at the corners of a 3 m × 2 m rectangle
MARKER_WORLD_POSITIONS: list[tuple[float, float]] = [
    (0.5, 0.5),   # marker 1: 0.5 m from left wall, 0.5 m from back wall
    (3.5, 0.5),   # marker 2
    (0.5, 2.5),   # marker 3
    (3.5, 2.5),   # marker 4
    # (2.0, 1.5), # optional 5th marker at room centre
]

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_NPZ  = OUTPUT_DIR / "homography_calibration.npz"
OUTPUT_THALG = OUTPUT_DIR / "homography_calibration.thalg"


# ===========================================================================
# Frame loading
# ===========================================================================

def _load_frame(recording_dir: Path, channel: int, frame_idx: int) -> np.ndarray:
    npz_path = recording_dir / f"ch{channel}_raw_data.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"NPZ file not found: {npz_path}\n"
            f"Check that RECORDING_DIR points to a session folder containing\n"
            f"ch0_raw_data.npz, ch1_raw_data.npz, ch2_raw_data.npz."
        )
    arr = np.load(npz_path)["frames"].astype(np.float32)
    if frame_idx >= arr.shape[0]:
        raise ValueError(
            f"FRAME_INDEX={frame_idx} is out of range "
            f"(recording has {arr.shape[0]} frames)."
        )
    return arr[frame_idx]


def _make_synthetic_frames(
    n_markers: int,
    marker_positions: list[tuple[float, float]],
    sensor=MLX90640,
) -> list[np.ndarray]:
    """Generate 3 synthetic calibration frames with visible hot-spots."""
    rng = np.random.default_rng(42)
    H, W = sensor.height, sensor.width
    frames = []

    # Map world marker positions to approximate pixel positions per camera
    # (simple linear scaling — for demo purposes only)
    world_xs = [p[0] for p in marker_positions]
    world_ys = [p[1] for p in marker_positions]
    w_min, w_max = min(world_xs), max(world_xs)
    h_min, h_max = min(world_ys), max(world_ys)

    cam_transforms = [
        lambda wx, wy: ((wx - w_min) / (w_max - w_min + 1e-6) * (W - 4) + 2,
                        (wy - h_min) / (h_max - h_min + 1e-6) * (H - 4) + 2),
        lambda wx, wy: ((1 - (wx - w_min) / (w_max - w_min + 1e-6)) * (W - 4) + 2,
                        (wy - h_min) / (h_max - h_min + 1e-6) * (H - 4) + 2),
        lambda wx, wy: ((wx - w_min) / (w_max - w_min + 1e-6) * (W - 4) + 2,
                        (1 - (wy - h_min) / (h_max - h_min + 1e-6)) * (H - 4) + 2),
    ]

    for cam_id in range(3):
        data = (rng.standard_normal((H, W)) * 0.3 + 22.0).astype(np.float32)
        transform = cam_transforms[cam_id]
        for wx, wy in marker_positions:
            px, py = transform(wx, wy)
            px, py = int(round(px)), int(round(py))
            px = max(1, min(W - 2, px))
            py = max(1, min(H - 2, py))
            data[py - 1:py + 2, px - 1:px + 2] = 50.0  # hot marker
        frames.append(data)

    return frames


# ===========================================================================
# Interactive clicking
# ===========================================================================

def _collect_clicks_for_camera(
    frame: np.ndarray,
    cam_id: int,
    marker_world_positions: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """Show a camera frame and collect N clicks, one per marker.

    Returns a list of (u, v) pixel clicks in the same order as
    marker_world_positions, or None if the user closed the window early.
    """
    n = len(marker_world_positions)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title(f"Camera {cam_id} — click markers in order")

    vmin, vmax = np.percentile(frame, [2, 98])
    ax.imshow(frame, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_title(
        f"Camera {cam_id}  —  click {n} heated markers in order\n"
        + "  ".join(f"{i+1}: ({wx:.2f}, {wy:.2f}) m"
                    for i, (wx, wy) in enumerate(marker_world_positions)),
        fontsize=9,
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Cross-hair guide: plot placeholder points to be updated as user clicks
    clicked_pts = []
    scatter = ax.scatter([], [], c="cyan", s=60, zorder=5, marker="+")

    # Instruction overlay
    fig.text(0.5, 0.01, "Left-click to mark • Right-click / Backspace to undo • Close window to cancel",
             ha="center", fontsize=8, color="lightgray")

    plt.tight_layout()

    try:
        clicks = plt.ginput(n=n, timeout=-1, show_clicks=True)
    except Exception:
        clicks = []

    plt.close(fig)

    if len(clicks) < n:
        print(f"  Camera {cam_id}: only {len(clicks)} of {n} clicks received — aborting.")
        return None

    return [(float(u), float(v)) for u, v in clicks[:n]]


# ===========================================================================
# Reprojection validation
# ===========================================================================

def _reprojection_error(
    H: np.ndarray,
    pixel_clicks: list[tuple[float, float]],
    world_positions: list[tuple[float, float]],
) -> tuple[float, float]:
    """Return (mean_error_m, max_error_m) for a set of correspondences."""
    from math import sqrt
    errors = []
    for (u, v), (xw_gt, yw_gt) in zip(pixel_clicks, world_positions):
        xw_pred, yw_pred = project_foot_point((u, v), H)
        err = sqrt((xw_pred - xw_gt) ** 2 + (yw_pred - yw_gt) ** 2)
        errors.append(err)
    return float(np.mean(errors)), float(np.max(errors))


def _show_validation_plot(
    frames: list[np.ndarray],
    all_clicks: list[list[tuple[float, float]]],
    homographies: HomographyMatrices,
    world_positions: list[tuple[float, float]],
) -> None:
    """3-panel validation: camera views + floor-plane reprojection."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    colours = plt.cm.tab10(np.linspace(0, 0.9, len(world_positions)))

    for cam_id in range(3):
        ax = axes[cam_id]
        frame = frames[cam_id]
        vmin, vmax = np.percentile(frame, [2, 98])
        ax.imshow(frame, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"Camera {cam_id}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        H = homographies[cam_id]
        for idx, ((u, v), (xw, yw)) in enumerate(
                zip(all_clicks[cam_id], world_positions)):
            ax.plot(u, v, "o", color=colours[idx], markersize=8, zorder=5)
            # Reproject world → pixel to show accuracy
            p_img = H @ np.array([u, v, 1.0])
            # (already in world coords via H; show reprojection arrow if noticeable)
            ax.annotate(
                f"M{idx+1}",
                (u, v), xytext=(u + 0.5, v - 1),
                color="white", fontsize=7, zorder=6,
            )

    # Floor map
    ax_floor = axes[3]
    ax_floor.set_facecolor("#111")
    ax_floor.set_title("Floor reprojection (world coords)", fontsize=9)
    ax_floor.set_xlabel("X (m)"); ax_floor.set_ylabel("Y (m)")

    # Plot ground-truth marker positions
    for idx, (xw, yw) in enumerate(world_positions):
        ax_floor.plot(xw, yw, "D", color=colours[idx], markersize=10,
                      zorder=5, label=f"M{idx+1} GT")

    # Plot reprojected positions from each camera
    cam_markers = ["^", "s", "o"]
    for cam_id in range(3):
        H = homographies[cam_id]
        for idx, (u, v) in enumerate(all_clicks[cam_id]):
            xw_pred, yw_pred = project_foot_point((u, v), H)
            ax_floor.plot(xw_pred, yw_pred, cam_markers[cam_id],
                          color=colours[idx], markersize=6, alpha=0.7,
                          label=f"Cam{cam_id}" if idx == 0 else "")

    ax_floor.legend(loc="lower right", fontsize=7, ncol=2)
    ax_floor.set_aspect("equal")
    ax_floor.grid(True, alpha=0.3)

    fig.suptitle("Homography Calibration Validation", fontsize=11)
    plt.tight_layout()

    val_path = OUTPUT_DIR / "homography_validation.png"
    fig.savefig(val_path, dpi=130, bbox_inches="tight", facecolor="#222")
    print(f"  Validation plot saved to {val_path}")
    plt.show()


# ===========================================================================
# Save / load helpers
# ===========================================================================

def _save_calibration(
    homographies: HomographyMatrices,
    all_clicks: list[list[tuple[float, float]]],
    world_positions: list[tuple[float, float]],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. NPZ — easy to inspect and load manually
    np.savez(
        OUTPUT_NPZ,
        h1=homographies.h1,
        h2=homographies.h2,
        h3=homographies.h3,
        world_positions=np.array(world_positions),
        clicks_cam0=np.array(all_clicks[0]),
        clicks_cam1=np.array(all_clicks[1]),
        clicks_cam2=np.array(all_clicks[2]),
    )
    print(f"  Calibration matrices saved to {OUTPUT_NPZ}")

    # 2. .thalg checkpoint — load directly into GeometricContactDetector
    detector = GeometricContactDetector(
        homography=homographies, epsilon_m=0.4, delta_m=0.5
    ).fit([])
    detector.save(OUTPUT_THALG)
    print(f"  Checkpoint saved to  {OUTPUT_THALG}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("  Hot-Point Homography Calibration")
    print("  Smart Thermal System — § 4.4.3.1")
    print("=" * 60)

    n_markers = len(MARKER_WORLD_POSITIONS)
    print(f"\nMarkers ({n_markers}):")
    for i, (xw, yw) in enumerate(MARKER_WORLD_POSITIONS):
        print(f"  M{i+1}: ({xw:.2f} m, {yw:.2f} m)")

    if n_markers < 4:
        print("\n[ERROR] At least 4 marker positions are required for a unique homography.")
        print("Add more entries to MARKER_WORLD_POSITIONS and re-run.")
        return

    # ── Load frames ─────────────────────────────────────────────────────────
    if RECORDING_DIR is not None:
        print(f"\nLoading frames from {RECORDING_DIR} (frame {FRAME_INDEX}) ...")
        try:
            frames = [
                _load_frame(RECORDING_DIR, cam_id, FRAME_INDEX)
                for cam_id in range(3)
            ]
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return
        print("  Frames loaded.")
    else:
        print("\n[DEMO MODE] RECORDING_DIR is None — using synthetic frames.")
        print("  Set RECORDING_DIR at the top of this script to use real data.\n")
        frames = _make_synthetic_frames(n_markers, MARKER_WORLD_POSITIONS,
                                        SENSOR_PROFILE)

    # ── Interactive clicking — one camera at a time ──────────────────────────
    all_clicks: list[list[tuple[float, float]]] = []
    homography_inputs: list[tuple[int, list]] = []

    print("\nFor each camera view:")
    print(f"  • Click each of the {n_markers} heated markers IN ORDER (M1, M2, ...)")
    print("  • Right-click or Backspace to undo the last click")
    print("  • Close the window to cancel calibration\n")

    for cam_id in range(3):
        print(f"Camera {cam_id}: click {n_markers} markers ...")
        clicks = _collect_clicks_for_camera(
            frames[cam_id], cam_id, MARKER_WORLD_POSITIONS
        )
        if clicks is None:
            print("\nCalibration cancelled.")
            return
        all_clicks.append(clicks)

        # Build correspondences for this camera
        pairs = list(zip(clicks, MARKER_WORLD_POSITIONS))
        homography_inputs.append((cam_id, pairs))
        print(f"  Camera {cam_id}: {n_markers} clicks recorded.")

    # ── Solve homographies ───────────────────────────────────────────────────
    print("\nSolving homography matrices (DLT + SVD) ...")
    try:
        homographies = solve_homography_from_markers(homography_inputs)
    except Exception as e:
        print(f"[ERROR] Homography solver failed: {e}")
        print("Check that markers are not collinear and all clicks are in the correct order.")
        return

    # ── Report reprojection errors ───────────────────────────────────────────
    print("\nReprojection errors:")
    all_ok = True
    for cam_id in range(3):
        H = homographies[cam_id]
        mean_err, max_err = _reprojection_error(
            H, all_clicks[cam_id], MARKER_WORLD_POSITIONS
        )
        status = "✓" if mean_err < 0.15 else "⚠"
        print(f"  Camera {cam_id}: mean={mean_err*100:.1f} cm  max={max_err*100:.1f} cm  {status}")
        if mean_err >= 0.20:
            all_ok = False
            print(f"    [WARN] Mean error > 20 cm — check marker positions or re-click camera {cam_id}.")

    if not all_ok:
        ans = input("\nErrors are high. Save anyway? [y/N] ").strip().lower()
        if ans != "y":
            print("Calibration not saved.")
            return

    # ── Validate visually ────────────────────────────────────────────────────
    print("\nShowing validation plot ...")
    _show_validation_plot(frames, all_clicks, homographies, MARKER_WORLD_POSITIONS)

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\nSaving calibration ...")
    _save_calibration(homographies, all_clicks, MARKER_WORLD_POSITIONS)

    # ── Usage reminder ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Calibration complete!  To use:")
    print()
    print("  import numpy as np")
    print("  from thermal_algorithms.core.types import HomographyMatrices")
    print("  from thermal_algorithms.contact_detection.geometric import \\")
    print("      GeometricContactDetector")
    print()
    print(f"  cal = np.load('{OUTPUT_NPZ}')")
    print("  H = HomographyMatrices(h1=cal['h1'], h2=cal['h2'], h3=cal['h3'])")
    print("  detector = GeometricContactDetector(homography=H).fit([])")
    print("=" * 60)


if __name__ == "__main__":
    main()
