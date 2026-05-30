"""Train HOG+SVM on real labelled data and compare visually with the
classical AdaptiveThresholdDetector (Section 4.4.2.2 vs 4.4.2.1).

Pipeline:
    1. Build a DatasetIndex over the project's recording root.
    2. Train HOGSVMDetector on every labelled (Frame, list[Detection]) example.
    3. Pick a few frames from `personpresence` (not in the training set) and
       run both detectors. Visualize side by side.

Run:
    python examples/demo_hog_svm.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset


DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIB_DIR = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "hog_svm_demo.png"


def overlay_boxes(ax, dets, color, label_field=None):
    for d in dets:
        x, y, w, h = d.bbox
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), w, h, linewidth=1.6,
            edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        if label_field is not None:
            txt = f"{d.score:.2f}"
            ax.text(x, y - 0.4, txt, color=color, fontsize=7)


def main() -> None:
    # --- 1. Train HOG+SVM on every labelled example ---
    idx = DatasetIndex(DATASET_ROOT, sensor_profile=MLX90640)
    train_ds = FrameLevelDataset(idx, class_filter=[1])
    print(f"DatasetIndex: {idx}")
    print(f"Training examples: {len(train_ds)}")

    hog_det = HOGSVMDetector(MLX90640, score_threshold=0.5)
    print(f"HOG window: {hog_det.window_size}, cell: {hog_det.cell_size}")
    hog_det.fit(list(train_ds))
    print(f"HOG fitted: feature_dim={hog_det.feature_dim}")

    # --- 2. Classical baseline (Tateno + Adaptive Threshold) for comparison ---
    pipeline = TatenoPipeline(sensor_profile=MLX90640)
    # Use empty-room calibration
    from thermal_algorithms.core.io import load_recording  # noqa
    calib_frames = list(idx.find("emptyroomwithpcscreen").load_frames(channel=1))
    # That returned an ndarray; build Frames manually
    empty_session = idx.find("emptyroomwithpcscreen")
    calib_arr = empty_session.load_frames(channel=1)
    from thermal_algorithms.core.types import Frame
    calib_frames = [
        Frame(data=calib_arr[i], timestamp=i / empty_session.fps, camera_id=1)
        for i in range(calib_arr.shape[0])
    ]
    pipeline.fit(calib_arr)
    cls_det = AdaptiveThresholdDetector(
        sensor_profile=MLX90640, c_offset=0.4,
        pixel_value_bounds=(1.0, 20.0),
    ).fit([])

    # --- 3. Choose test frames from a recording NOT in the training set ---
    test_session = idx.find("personpresence")
    test_frames_arr = test_session.load_frames(channel=1)
    test_frame_indices = [40, 80, 120]   # mid-recording, person present
    test_frames = [
        Frame(data=test_frames_arr[i], timestamp=i / test_session.fps, camera_id=1)
        for i in test_frame_indices
    ]

    # --- 4. Plot raw + classical-residual-detections + HOG detections ---
    fig, axes = plt.subplots(len(test_frames), 3, figsize=(9, 3.2 * len(test_frames)))
    if len(test_frames) == 1:
        axes = np.array([axes])

    for i, raw in enumerate(test_frames):
        vmin, vmax = np.percentile(raw.data, [2, 98])

        # Run classical pipeline (Tateno → AdaptiveThreshold)
        residual = pipeline.predict(raw)
        classical_dets = cls_det.predict(residual)

        # Run HOG+SVM directly on raw frame
        hog_dets = hog_det.predict(raw)

        # Plot 1: raw + classical bboxes
        ax = axes[i, 0]
        ax.imshow(raw.data, cmap="inferno", vmin=vmin, vmax=vmax)
        overlay_boxes(ax, classical_dets, "lime")
        ax.set_title(f"AdaptiveThreshold (Sect 4.4.2.1)\nframe {test_frame_indices[i]}: {len(classical_dets)} dets")

        # Plot 2: raw + HOG bboxes
        ax = axes[i, 1]
        ax.imshow(raw.data, cmap="inferno", vmin=vmin, vmax=vmax)
        overlay_boxes(ax, hog_dets, "cyan", label_field="score")
        ax.set_title(f"HOG+SVM (Sect 4.4.2.2)\nframe {test_frame_indices[i]}: {len(hog_dets)} dets")

        # Plot 3: residual for reference
        ax = axes[i, 2]
        ax.imshow(residual.data, cmap="viridis", vmin=0)
        ax.set_title(f"Tateno residual\npeak={residual.data.max():.1f}")

        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Section 4.4.2 comparison — classical vs HOG+SVM on personpresence",
        y=1.02,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
