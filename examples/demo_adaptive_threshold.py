"""Visual demo: AdaptiveThresholdDetector on real thermal data (§ 4.4.2.1).

Calibrates the Tateno preprocessing pipeline on an empty-room recording, then
runs preprocessing → adaptive thresholding on frames from `personpresence`
and `3ppl`. Visualizes:

    raw frame  →  preprocessed residual  →  segmentation mask
              →  post-morphology mask    →  detections overlay

Run from the project root:
    python examples/demo_adaptive_threshold.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Frame
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline


DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIB_DIR = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "adaptive_threshold_demo.png"

CHANNEL = 1


def load_npz_as_frames(path: Path, camera_id: int) -> list[Frame]:
    data = np.load(path)["frames"]
    return [
        Frame(data=data[i].astype(np.float32), timestamp=float(i) / 8.0,
              camera_id=camera_id)
        for i in range(data.shape[0])
    ]


def find_scene_for_session(session_dir: Path, pipeline: TatenoPipeline,
                           detector: AdaptiveThresholdDetector,
                           min_detections: int = 1) -> tuple[Frame, list]:
    """Scan a recording for a frame where the detector finds >= min_detections."""
    frames = load_npz_as_frames(session_dir / f"ch{CHANNEL}_raw_data.npz", CHANNEL)
    for f in frames[20:]:  # skip warmup
        residual = pipeline.predict(f)
        dets = detector.predict(residual)
        if len(dets) >= min_detections:
            return f, dets
    # Fall back to mid-recording
    f = frames[len(frames) // 2]
    return f, detector.predict(pipeline.predict(f))


def plot_pipeline(ax_row, raw, residual, mask_raw, mask_closed, dets, title):
    """Plot 5 panels for one scene: raw, residual, raw mask, closed mask, overlay."""
    vmin, vmax = np.percentile(raw.data, [2, 98])

    ax_row[0].imshow(raw.data, cmap="inferno", vmin=vmin, vmax=vmax)
    ax_row[0].set_title(f"{title}\nraw frame")

    ax_row[1].imshow(residual.data, cmap="viridis", vmin=0)
    ax_row[1].set_title(f"residual\npeak={residual.data.max():.1f}")

    ax_row[2].imshow(mask_raw, cmap="gray", vmin=0, vmax=255)
    ax_row[2].set_title(f"mask after\nadaptive threshold")

    ax_row[3].imshow(mask_closed, cmap="gray", vmin=0, vmax=255)
    ax_row[3].set_title("mask after\nmorph. closing")

    ax_row[4].imshow(raw.data, cmap="inferno", vmin=vmin, vmax=vmax)
    for d in dets:
        x, y, w, h = d.bbox
        rect = patches.Rectangle((x - 0.5, y - 0.5), w, h, linewidth=1.6,
                                 edgecolor="lime", facecolor="none")
        ax_row[4].add_patch(rect)
        mean_t = d.thermal_features["mean_temp"] if d.thermal_features else 0.0
        ax_row[4].text(x, y - 0.4, f"{mean_t:.1f}", color="lime", fontsize=7)
    ax_row[4].set_title(f"detections ({len(dets)})")

    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])


def main() -> None:
    # 1. Fit Tateno preprocessing on empty-room calibration data.
    calib_frames = load_npz_as_frames(CALIB_DIR / f"ch{CHANNEL}_raw_data.npz", CHANNEL)
    pipeline = TatenoPipeline(sensor_profile=MLX90640).fit(calib_frames)
    print(f"Tateno fitted on {len(calib_frames)} calibration frames")

    # 2. Build the detector. Since input is a residual (above-background °C),
    #    use a residual-appropriate intensity bound.
    detector = AdaptiveThresholdDetector(
        sensor_profile=MLX90640,
        c_offset=0.4,
        min_solidity=0.45,
        min_area_pixels=4,
        pixel_value_bounds=(1.0, 20.0),  # residual °C — human anomalies
    ).fit([])
    print(f"Detector: block_size={detector.block_size}, c_offset={detector._c_offset}")

    # 3. Pick two recordings of interest.
    scenes: list[tuple[str, Path]] = [
        ("Single person", DATASET_ROOT / "personpresence" / "20260405_195423"),
        ("Three people",  DATASET_ROOT / "3ppl"          / "20260405_204500"),
    ]
    # Resolve the "3ppl" subfolder dynamically — sessions are timestamped.
    three_ppl_root = DATASET_ROOT / "3ppl"
    if not scenes[1][1].exists() and three_ppl_root.exists():
        sub = sorted(three_ppl_root.iterdir())
        if sub:
            scenes[1] = ("Three people", sub[0])

    fig, axes = plt.subplots(len(scenes), 5, figsize=(15, 3.2 * len(scenes)))
    if len(scenes) == 1:
        axes = np.array([axes])

    for i, (label, session_dir) in enumerate(scenes):
        if not session_dir.exists():
            print(f"  SKIP {label} — directory {session_dir} not found")
            continue
        raw_frame, dets = find_scene_for_session(session_dir, pipeline, detector,
                                                 min_detections=1)
        residual = pipeline.predict(raw_frame)
        mask_raw = detector._segment(residual.data)
        mask_closed = detector._close(mask_raw)
        print(f"  {label}: t={raw_frame.timestamp:.2f}s  →  {len(dets)} detections")
        for d in dets:
            print(f"      bbox={d.bbox}  mean_temp={d.thermal_features['mean_temp']:.2f} "
                  f"area={d.thermal_features['area_pixels']:.0f}px")

        plot_pipeline(axes[i], raw_frame, residual, mask_raw, mask_closed, dets, label)

    fig.suptitle(
        "AdaptiveThresholdDetector (§ 4.4.2.1) — Tateno preprocessing → "
        "adaptive threshold → morph closing → geometric filtering",
        y=1.02,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
