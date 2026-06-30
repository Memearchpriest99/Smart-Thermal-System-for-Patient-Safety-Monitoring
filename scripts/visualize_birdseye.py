"""Bird's-eye visualization of person position from a manual homography.

Takes a ``homography_calibration.npz`` produced by the image annotator's
calibration dialog, runs the project's person detector on each camera of a
session, projects every foot-point onto the calibrated floor plane, fuses the
three views, and renders a top-down "bird's-eye" animation showing where the
person is.

Output
------
    reports/birdseye_<session>.gif   — animated: 3 camera views + floor plane
    reports/birdseye_<session>.png   — static trajectory over the whole clip

Run
---
    python scripts/visualize_birdseye.py
    python scripts/visualize_birdseye.py --session datasets/waveshare_work/calibrate_room
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import Frame, HomographyMatrices
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.contact_detection.multi_view.homography import (
    project_foot_point,
)
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections

# Same person-detector tuning the annotator's recommender uses.
PERSON_DET_KWARGS = dict(
    c_offset=0.5, morph_kernel_size=9, min_area_pixels=60, min_solidity=0.30,
    min_aspect_ratio=0.10, max_aspect_ratio=8.0,
    pixel_value_bounds=(1.5, 1000.0),
)
CAM_COLORS = ["#ff5252", "#448aff", "#69f0ae"]   # ch0 / ch1 / ch2
CAM_MARKERS = ["^", "s", "o"]
EPSILON = 0.5          # fusion threshold, in floor (rectangle) units
TRAIL_LEN = 25         # fused-position trail length
FPS = 8


def _load_cube(session: Path, ch: int) -> np.ndarray:
    return np.load(session / f"ch{ch}_raw_data.npz")["frames"].astype(np.float32)


def _fit_backgrounds(root: Path, candidates=("empty_room", "calibrate_room")):
    """Fit a per-channel Tateno background from the first sibling that has NPZs."""
    for name in candidates:
        bg_dir = root / name
        if not bg_dir.is_dir():
            continue
        pre, ok = {}, True
        for ch in range(3):
            p = bg_dir / f"ch{ch}_raw_data.npz"
            if not p.is_file():
                ok = False
                break
            tp = TatenoPipeline(WAVESHARE_26984)
            tp.fit(np.load(p)["frames"].astype(np.float32))
            pre[ch] = tp
        if ok:
            return pre, name
    raise SystemExit("No empty_room / calibrate_room background NPZs found.")


def _detect_all(session: Path, bg_root: Path):
    """Per-frame detections + projected foot-points for all three cameras."""
    cal = np.load(session / "homography_calibration.npz")
    H = HomographyMatrices(h1=cal["h1"], h2=cal["h2"], h3=cal["h3"])
    rect = cal["rect_wh"] if "rect_wh" in cal.files else np.array([1.0, 1.0])
    corners = cal["world_positions"][:4] if "world_positions" in cal.files else \
        np.array([[0, 0], [rect[0], 0], rect, [0, rect[1]]])

    pre, bg_name = _fit_backgrounds(bg_root)
    det = AdaptiveThresholdDetector(WAVESHARE_26984, **PERSON_DET_KWARGS)
    cubes = [_load_cube(session, ch) for ch in range(3)]
    n = min(c.shape[0] for c in cubes)

    per_frame = []   # list of dicts: {ch: [(bbox, foot, (X,Y)), ...]}
    fused_pts = []   # list of [(X,Y), ...] fused actor positions
    next_id = 0
    for i in range(n):
        dets_per_cam = []
        frame_info = {}
        for ch in range(3):
            res = pre[ch].predict(
                Frame(data=cubes[ch][i], timestamp=0.0, camera_id=ch))
            dets = det.predict(res)
            dets_per_cam.append(dets)
            frame_info[ch] = [
                (d.bbox, d.foot_point, project_foot_point(d.foot_point, H[ch]))
                for d in dets
            ]
        actors, next_id = fuse_detections(
            dets_per_cam, H, epsilon_m=EPSILON, next_track_id=next_id)
        per_frame.append(frame_info)
        fused_pts.append([a.world_xy for a in actors])

    return cubes, per_frame, fused_pts, corners, bg_name, n, rect


def _floor_window(corners: np.ndarray, fused_pts) -> tuple[float, float, float, float]:
    """A bird's-eye axis window: the rectangle plus a margin, clipped so wild
    near-horizon projections don't blow up the scale."""
    xs = list(corners[:, 0])
    ys = list(corners[:, 1])
    for pts in fused_pts:
        for (x, y) in pts:
            if abs(x) < 5 and abs(y) < 5:
                xs.append(x); ys.append(y)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mx = 0.35 * max(x1 - x0, 1e-3)
    my = 0.35 * max(y1 - y0, 1e-3)
    return x0 - mx, x1 + mx, y0 - my, y1 + my


def _draw_floor(ax, corners, xlim, ylim, unit="rect units"):
    poly = np.vstack([corners, corners[0]])
    ax.plot(poly[:, 0], poly[:, 1], "-", color="#888", lw=1.5, zorder=2)
    ax.fill(corners[:, 0], corners[:, 1], color="#2a2a2a", alpha=0.5, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])   # invert Y → image-like top-down
    ax.set_aspect("equal")
    ax.set_facecolor("#111")
    ax.set_title("Bird's-eye floor plane", fontsize=10, color="white")
    ax.set_xlabel(f"X [{unit}]", fontsize=8, color="#aaa")
    ax.set_ylabel(f"Y [{unit}]", fontsize=8, color="#aaa")
    ax.tick_params(colors="#777", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#444")


def render(session: Path, bg_root: Path, out_dir: Path,
           meters: tuple[float, float] | None = None) -> None:
    cubes, per_frame, fused_pts, corners, bg_name, n, rect = _detect_all(
        session, bg_root)
    scene = session.name

    # Optional APPROXIMATE metric relabelling (display-only): rescale the
    # consistent floor plane so axes read in ballpark metres. This does NOT
    # change the homography or fusion — it only stretches plotted coordinates,
    # and is only meaningful insofar as the clicked "rectangle" matched a real
    # W×H on the floor.
    unit = "rect units"
    sx = sy = 1.0
    if meters is not None:
        sx = float(meters[0]) / float(rect[0])
        sy = float(meters[1]) / float(rect[1])
        corners = corners * np.array([sx, sy])
        fused_pts = [[(x * sx, y * sy) for (x, y) in pts] for pts in fused_pts]
        unit = "m, approx"
        print(f"[birdseye] approx metric rescale: rect {rect} units -> "
              f"{meters[0]}x{meters[1]} m  (sx={sx:.3f}, sy={sy:.3f})")

    def sp(xy):   # scale a projected point for display
        return (xy[0] * sx, xy[1] * sy)

    xlim = _floor_window(corners, fused_pts)[:2]
    ylim = _floor_window(corners, fused_pts)[2:]
    print(f"[birdseye] {scene}: {n} frames, background='{bg_name}', "
          f"frames with a fused person: {sum(1 for p in fused_pts if p)}")

    # ── animated GIF ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3))
    fig.patch.set_facecolor("#1a1a1a")
    images = []
    trail: list[tuple[float, float]] = []

    for i in range(n):
        for ax in axes:
            ax.clear()
        # camera panels
        for ch in range(3):
            ax = axes[ch]
            frame = cubes[ch][i]
            vmin, vmax = np.percentile(frame, [2, 99])
            ax.imshow(frame, cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"Ch {ch}", fontsize=10, color=CAM_COLORS[ch])
            ax.set_xticks([]); ax.set_yticks([])
            for (x, y, w, h), foot, _ in per_frame[i][ch]:
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                           edgecolor=CAM_COLORS[ch], lw=1.5))
                ax.plot(foot[0], foot[1], "o", color=CAM_COLORS[ch], ms=4)

        # bird's-eye panel
        axf = axes[3]
        _draw_floor(axf, corners, xlim, ylim, unit)
        for ch in range(3):
            for _, _, proj in per_frame[i][ch]:
                X, Y = sp(proj)
                axf.plot(X, Y, CAM_MARKERS[ch], color=CAM_COLORS[ch],
                         ms=7, alpha=0.65, zorder=3)
        # fused person + trail
        if fused_pts[i]:
            cx = float(np.mean([p[0] for p in fused_pts[i]]))
            cy = float(np.mean([p[1] for p in fused_pts[i]]))
            trail.append((cx, cy))
            trail[:] = trail[-TRAIL_LEN:]
            for actor in fused_pts[i]:
                axf.plot(actor[0], actor[1], "*", color="#ffd740", ms=20,
                         markeredgecolor="black", zorder=5)
        if len(trail) > 1:
            t = np.array(trail)
            axf.plot(t[:, 0], t[:, 1], "-", color="#ffd740", lw=1.2,
                     alpha=0.5, zorder=4)
        axf.text(0.02, 0.98, f"frame {i + 1}/{n}", transform=axf.transAxes,
                 color="white", fontsize=8, va="top")

        fig.suptitle(f"Bird's-eye person tracking — {scene}",
                     color="white", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        images.append(Image.fromarray(buf).convert("RGB"))

    plt.close(fig)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"birdseye_{scene}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"[birdseye] wrote {gif_path}  ({len(images)} frames)")

    # ── static trajectory ────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.patch.set_facecolor("#1a1a1a")
    _draw_floor(ax2, corners, xlim, ylim, unit)
    ax2.set_title(f"Person trajectory — {scene}", fontsize=11, color="white")
    pts = [(i, p[0], p[1]) for i, frame_pts in enumerate(fused_pts)
           for p in frame_pts]
    if pts:
        arr = np.array(pts)
        sc = ax2.scatter(arr[:, 1], arr[:, 2], c=arr[:, 0], cmap="viridis",
                         s=40, zorder=5, edgecolor="black", linewidth=0.3)
        cb = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label("frame index", color="white")
        cb.ax.tick_params(colors="white")
    png_path = out_dir / f"birdseye_{scene}.png"
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=130, facecolor="#1a1a1a")
    plt.close(fig2)
    print(f"[birdseye] wrote {png_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path,
                    default=Path("datasets/waveshare_work/calibrate_room"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--meters", type=float, nargs=2, metavar=("W", "H"),
                    default=None,
                    help="Approximate real rectangle size (m) for a metric "
                         "axis relabel; e.g. --meters 2.5 3.5")
    args = ap.parse_args()

    session = args.session
    if not (session / "homography_calibration.npz").is_file():
        raise SystemExit(f"No homography_calibration.npz in {session} — "
                         "calibrate it in the annotator first.")
    render(session, session.parent, args.out, meters=args.meters)


if __name__ == "__main__":
    main()
