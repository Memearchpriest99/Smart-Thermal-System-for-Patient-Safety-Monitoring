"""Visual demo: OtsuFireDetector on real fire/lighter recordings (§ 4.4.4).

Runs the full three-stage fire detection pipeline on a lighter recording:

    raw frame → variance gate → Otsu segmentation → morphological shaping
             → DTC classifier (Ignition Source / Potential Fire / Safe)
             → temporal mass-gradient tracker → Active Combustion?

Visualizes two panels:
  1. Per-frame grid: segmentation mask + detected fire level annotated on
     the raw thermal image for every Nth frame.
  2. Time-series: peak temperature and fire-level state over the entire clip.

Run from the project root:
    python examples/demo_fire_detection.py

Output: outputs/fire_detection_demo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import FireLevel, Frame
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline


# ---------------------------------------------------------------------------
# Configuration — adjust to your dataset layout
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIB_DIR    = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"

# Fire / lighter recordings from Table 4 of the report.
# The script picks the first one that exists on disk.
FIRE_SCENE_CANDIDATES = [
    DATASET_ROOT / "lighter_on_3_sec",
    DATASET_ROOT / "lighter_on_and_off_3_sec",
    DATASET_ROOT / "lighter_on_off_on_off_5_sec",
]

CHANNEL       = 1     # channel to use (0 / 1 / 2)
FRAME_STRIDE  = 4     # show every Nth frame in the grid panel
N_GRID_FRAMES = 8     # how many frames to show in the grid
OUTPUT_PATH   = Path(__file__).parent.parent / "outputs" / "fire_detection_demo.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(npz_path: Path, camera_id: int) -> list[Frame]:
    arr = np.load(npz_path)["frames"].astype(np.float32)
    return [Frame(data=arr[i], timestamp=i / 8.0, camera_id=camera_id)
            for i in range(arr.shape[0])]


def find_session(scene_root: Path) -> Path | None:
    """Return the first session subfolder (or scene_root itself) that has data."""
    npz = scene_root / f"ch{CHANNEL}_raw_data.npz"
    if npz.is_file():
        return scene_root
    for sub in sorted(scene_root.iterdir()):
        if (sub / f"ch{CHANNEL}_raw_data.npz").is_file():
            return sub
    return None


LEVEL_COLOUR = {
    FireLevel.SAFE:              "white",
    FireLevel.POTENTIAL_FIRE:    "yellow",
    FireLevel.IGNITION_SOURCE:   "orange",
    FireLevel.ACTIVE_COMBUSTION: "red",
}
LEVEL_SHORT = {
    FireLevel.SAFE:              "SAFE",
    FireLevel.POTENTIAL_FIRE:    "PF",
    FireLevel.IGNITION_SOURCE:   "IGN",
    FireLevel.ACTIVE_COMBUSTION: "COMB",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Find a fire recording ────────────────────────────────────────────
    session_dir = None
    for candidate in FIRE_SCENE_CANDIDATES:
        if candidate.exists():
            session_dir = find_session(candidate)
            if session_dir:
                break

    if session_dir is None:
        print("No fire recording found under DATASET_ROOT.  Generating synthetic data.")
        session_dir = None

    # ── 2. Calibrate Tateno preprocessor ────────────────────────────────────
    calib_npz = CALIB_DIR / f"ch{CHANNEL}_raw_data.npz"
    if calib_npz.is_file():
        calib_frames = load_frames(calib_npz, CHANNEL)
        preprocessor = TatenoPipeline(MLX90640).fit(calib_frames)
        print(f"Tateno fitted on {len(calib_frames)} calibration frames  "
              f"(σ={preprocessor.sigma:.2f}px)")
    else:
        print("Calibration NPZ not found — using unfitted preprocessor (raw frames).")
        preprocessor = None

    # ── 3. Load test frames (or synthesise if none found) ───────────────────
    if session_dir is not None:
        frames = load_frames(session_dir / f"ch{CHANNEL}_raw_data.npz", CHANNEL)
        title_suffix = f"real data: {session_dir.parent.name}/{session_dir.name}"
        print(f"Loaded {len(frames)} frames from {session_dir}")
    else:
        # Synthetic fallback: hot blob appears at frame 10, grows until frame 25
        rng = np.random.default_rng(0)
        h, w = MLX90640.height, MLX90640.width
        frames = []
        for i in range(40):
            data = (rng.standard_normal((h, w)) * 0.5 + 25.0).astype(np.float32)
            if i >= 10:
                r = max(1, int((i - 9) * 0.4))
                data[h // 2 - r: h // 2 + r + 1, w // 2 - r: w // 2 + r + 1] = 65.0
            frames.append(Frame(data=data, timestamp=i / 8.0, camera_id=CHANNEL))
        title_suffix = "synthetic data (lighter ignition simulation)"
        print(f"Using {len(frames)} synthetic frames.")

    # ── 4. Run OtsuFireDetector on every frame ───────────────────────────────
    detector = OtsuFireDetector(
        sensor_profile=MLX90640,
        t_ign=45.0, t_fire=60.0,
        delta_t=1.0, tau_step=4, s_threshold=2, t_measure=8.0,
    ).fit([])

    alerts = []
    for frame in frames:
        inp = preprocessor.predict(frame) if preprocessor else frame
        alert = detector.predict(inp)
        alerts.append(alert)

    levels  = [a.level for a in alerts]
    peaks   = [float(f.data.max()) for f in frames]
    times   = [f.timestamp for f in frames]

    n_fire = sum(1 for lv in levels if lv != FireLevel.SAFE)
    print(f"\nResults over {len(frames)} frames:")
    for lv in FireLevel:
        cnt = sum(1 for l in levels if l == lv)
        bar = "█" * cnt
        print(f"  {lv.value:<22} {cnt:4d}  {bar}")

    # ── 5. Visualise ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(2, N_GRID_FRAMES, hspace=0.45, wspace=0.05,
                           height_ratios=[1, 0.7])

    # — Top row: frame grid —
    grid_indices = np.linspace(0, len(frames) - 1, N_GRID_FRAMES, dtype=int)
    for col, idx in enumerate(grid_indices):
        ax = fig.add_subplot(gs[0, col])
        frame = frames[idx]
        lv    = levels[idx]
        vmin, vmax = np.percentile(frame.data, [2, 98])
        ax.imshow(frame.data, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(
            f"t={frame.timestamp:.1f}s\n{LEVEL_SHORT[lv]}",
            fontsize=8,
            color=LEVEL_COLOUR[lv] if lv != FireLevel.SAFE else "white",
            pad=2,
        )
        # Highlight alarm frames with a coloured border
        if lv != FireLevel.SAFE:
            for spine in ax.spines.values():
                spine.set_edgecolor(LEVEL_COLOUR[lv])
                spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    # — Bottom: time-series —
    ax_ts = fig.add_subplot(gs[1, :])
    ax_ts.plot(times, peaks, color="tomato", linewidth=1.5, label="peak temp (°C)")

    # Shade regions by fire level
    level_y = {"SAFE": 0, "POTENTIAL_FIRE": 1, "IGNITION_SOURCE": 2, "ACTIVE_COMBUSTION": 3}
    level_vals = [level_y[lv.name] for lv in levels]
    colours = [LEVEL_COLOUR[lv] for lv in levels]
    for i in range(len(times) - 1):
        ax_ts.axvspan(times[i], times[i + 1], alpha=0.3, color=colours[i], linewidth=0)

    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Peak pixel temp (°C)")
    ax_ts.set_xlim(times[0], times[-1])
    # Legend patches
    legend_patches = [
        mpatches.Patch(color=LEVEL_COLOUR[lv], label=lv.value, alpha=0.7)
        for lv in FireLevel
    ]
    ax_ts.legend(handles=legend_patches, loc="upper left", fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    fig.suptitle(
        f"OtsuFireDetector (§ 4.4.4) — {title_suffix}\n"
        f"Frames: {len(frames)}  |  Non-SAFE: {n_fire}  |  "
        f"Sensor: {MLX90640.name} {MLX90640.width}×{MLX90640.height}",
        fontsize=11,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="#111111")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
