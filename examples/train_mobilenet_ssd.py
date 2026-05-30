"""Train MicroMobileNet-SSD on the labeled thermal data and produce a visual
demo comparing all three Section 4.4.2 detectors.

Designed to be run from your Windows machine inside a conda env with PyTorch
installed. The Linux sandbox the rest of the project runs in does not have
torch, so training happens locally and the checkpoint is saved into the
project's checkpoints/ directory.

Usage (from a conda env that has torch + the project's other deps installed):
    cd "E:\\Documents\\Claude\\Projects\\Final Project"
    python examples/train_mobilenet_ssd.py
        --dataset "C:\\Users\\Guy\\Desktop\\dataset"
        --epochs 50

Outputs:
    checkpoints/mobilenet_ssd_detector/MLX90640.thalg   - trained model
    outputs/mobilenet_ssd_demo.png                       - 3-way comparison plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.core.checkpoints import CheckpointRegistry
from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Frame
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def overlay_boxes(ax, dets, color, draw_scores=False):
    for d in dets:
        x, y, w, h = d.bbox
        rect = patches.Rectangle(
            (x - 0.5, y - 0.5), w, h, linewidth=1.6,
            edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        if draw_scores:
            ax.text(x, y - 0.4, f"{d.score:.2f}", color=color, fontsize=7)


def train(args):
    print(f"Dataset root: {args.dataset}")
    idx = DatasetIndex(args.dataset, sensor_profile=MLX90640)
    print(f"  {idx}")
    train_ds = FrameLevelDataset(idx, class_filter=[1])
    print(f"Training examples (humans, all channels): {len(train_ds)}")

    detector = MobileNetSSDDetector(
        sensor_profile=MLX90640,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        score_threshold=args.score_threshold,
        device=args.device,
        random_state=args.seed,
    )
    print(
        f"MobileNet-SSD: device={args.device}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}"
    )

    detector.fit(list(train_ds), verbose=True)

    # Register the checkpoint.
    registry = CheckpointRegistry(root=CHECKPOINTS_DIR)
    path = registry.register(detector)
    print(f"Saved checkpoint -> {path}")
    return detector, idx


def demo(detector_mlssd, idx, args):
    """Three-way visual comparison on the personpresence recording."""
    # Build the other two detectors for side-by-side comparison.
    pipeline = TatenoPipeline(sensor_profile=MLX90640)
    empty = idx.find("emptyroomwithpcscreen")
    pipeline.fit(empty.load_frames(channel=1))

    cls_det = AdaptiveThresholdDetector(
        sensor_profile=MLX90640, c_offset=0.4, pixel_value_bounds=(1.0, 20.0),
    ).fit([])

    train_ds = FrameLevelDataset(idx, class_filter=[1])
    hog_det = HOGSVMDetector(sensor_profile=MLX90640, score_threshold=0.5)
    hog_det.fit(list(train_ds))

    # Pick test frames from personpresence.
    test = idx.find("personpresence")
    arr = test.load_frames(channel=1)
    test_indices = [40, 80, 120]
    test_frames = [
        Frame(data=arr[i], timestamp=i / test.fps, camera_id=1)
        for i in test_indices
    ]

    fig, axes = plt.subplots(len(test_frames), 3, figsize=(9.5, 3.4 * len(test_frames)))
    if len(test_frames) == 1:
        axes = np.array([axes])

    for i, raw in enumerate(test_frames):
        vmin, vmax = np.percentile(raw.data, [2, 98])

        # Classical: needs Tateno residual + adaptive threshold
        residual = pipeline.predict(raw)
        cls_dets = cls_det.predict(residual)

        # HOG+SVM: operates on raw frame
        hog_dets = hog_det.predict(raw)

        # MobileNet-SSD: operates on raw frame
        ssd_dets = detector_mlssd.predict(raw)

        for col, (title, dets, color) in enumerate([
            ("AdaptiveThreshold (4.4.2.1)", cls_dets, "lime"),
            ("HOG+SVM (4.4.2.2)",          hog_dets, "cyan"),
            ("MobileNet-SSD (4.4.2.3)",    ssd_dets, "magenta"),
        ]):
            ax = axes[i, col]
            ax.imshow(raw.data, cmap="inferno", vmin=vmin, vmax=vmax)
            overlay_boxes(ax, dets, color, draw_scores=(col > 0))
            ax.set_title(f"{title}\nframe {test_indices[i]}: {len(dets)} dets")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Section 4.4.2 - three detectors on personpresence", y=1.02)
    fig.tight_layout()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "mobilenet_ssd_demo.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   help="Path to the dataset root (containing 2pplfight/, emptyroomwithpcscreen/, ...).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cpu | cuda | cuda:0 ... (auto if omitted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-demo", action="store_true",
                   help="Train and save checkpoint but skip the visual demo.")
    args = p.parse_args()

    detector, idx = train(args)
    if not args.skip_demo:
        demo(detector, idx, args)


if __name__ == "__main__":
    main()
