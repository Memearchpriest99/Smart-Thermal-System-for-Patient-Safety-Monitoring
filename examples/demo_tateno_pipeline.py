"""Visual sanity check for the Tateno preprocessing pipeline (§ 4.4.1).

Loads an empty-room recording for calibration, then runs an occupied-room
recording through the pipeline and visualizes:

    raw frame   →   background B(x,y)   →   smoothed   →   residual

Run from the project root:

    python examples/demo_tateno_pipeline.py

The output PNG is written to outputs/tateno_demo.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIBRATION_DIR = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"
TEST_DIR = DATASET_ROOT / "personpresence" / "20260405_195423"
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "tateno_demo.png"

CHANNEL = 1   # ch1 has cleaner statistics than ch0 (no stuck pixels)
TEST_FRAME_INDEX = 80  # mid-recording, person present


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz_as_frames(path: Path, camera_id: int) -> list[Frame]:
    """Convert an .npz raw_data file (frames: (N, H, W)) into a list of Frame."""
    data = np.load(path)["frames"]
    return [
        Frame(
            data=data[i].astype(np.float32),
            timestamp=float(i) / 8.0,  # assume 8 Hz acquisition
            camera_id=camera_id,
        )
        for i in range(data.shape[0])
    ]


def clip_outliers(arr: np.ndarray, low_pct=1, high_pct=99) -> tuple[float, float]:
    """Return (vmin, vmax) clipped at the given percentiles, for display only."""
    return float(np.percentile(arr, low_pct)), float(np.percentile(arr, high_pct))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load calibration (empty room) and a test frame (person present).
    calib_frames = load_npz_as_frames(
        CALIBRATION_DIR / f"ch{CHANNEL}_raw_data.npz",
        camera_id=CHANNEL,
    )
    test_frames = load_npz_as_frames(
        TEST_DIR / f"ch{CHANNEL}_raw_data.npz",
        camera_id=CHANNEL,
    )
    print(f"Calibration: {len(calib_frames)} frames from {CALIBRATION_DIR.name}/ch{CHANNEL}")
    print(f"Test:        {len(test_frames)} frames from {TEST_DIR.name}/ch{CHANNEL}")

    # 2. Fit the pipeline on the empty-room frames.
    pipeline = TatenoPipeline(sensor_profile=MLX90640)
    pipeline.fit(calib_frames)
    print(
        f"Fitted. sigma={pipeline.sigma:.2f}px, kernel={pipeline.kernel_size}, "
        f"background mean={pipeline.background.mean():.2f} °C"
    )

    # 3. Run a test frame through the pipeline.
    test_frame = test_frames[TEST_FRAME_INDEX]
    residual = pipeline.predict(test_frame)
    print(
        f"Test frame range: [{test_frame.data.min():.1f}, {test_frame.data.max():.1f}] °C  →  "
        f"residual range: [{residual.data.min():.2f}, {residual.data.max():.2f}]"
    )

    # 4. Plot raw / background / residual side by side.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    vmin_raw, vmax_raw = clip_outliers(test_frame.data, 2, 98)
    im0 = axes[0].imshow(test_frame.data, cmap="inferno", vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_title(
        f"Raw frame (ch{CHANNEL}, t={test_frame.timestamp:.2f}s)\n"
        f"range [{vmin_raw:.1f}, {vmax_raw:.1f}] °C"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="°C")

    vmin_bg, vmax_bg = clip_outliers(pipeline.background, 2, 98)
    im1 = axes[1].imshow(pipeline.background, cmap="inferno", vmin=vmin_bg, vmax=vmax_bg)
    axes[1].set_title(
        f"Learned background B(x,y)\n"
        f"from {len(calib_frames)} empty-room frames"
    )
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="°C")

    im2 = axes[2].imshow(residual.data, cmap="viridis", vmin=0)
    axes[2].set_title(
        "Rectified residual |I_s - B|\n"
        f"peak = {residual.data.max():.2f} °C"
    )
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="°C")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Tateno Preprocessing Pipeline (§ 4.4.1) — Real Dataset Demo", y=1.02)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
