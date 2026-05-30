"""End-to-end ThermalPipeline demo (§ 4.2.1 flow diagram).

Wires together the full system matching Figure 5 of the Engineering Report:

    Thermal frames (3 cameras)
        ├── Preprocessing (Tateno)
        ├── Human detection (AdaptiveThreshold)     ─► feeds contact path
        ├── Fire detection (OtsuFireDetector)        ─► FIRE alert
        └── Contact detection (GeometricFusion)      ─► CONTACT alert

Runs on a real recording when available; otherwise generates a synthetic
demonstration sequence that triggers a fire alert mid-clip.

Visualises:
  • Top row    — three camera views from the triggering frame
  • Middle row — per-camera preprocessed frames + fire level badges
  • Bottom     — alert timeline over the clip (FIRE / CONTACT / SAFE)

Run from the project root:
    python examples/demo_pipeline.py

Output: outputs/pipeline_demo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Frame
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
from thermal_algorithms.human_detection.adaptive_threshold import (
    AdaptiveThresholdDetector,
)
from thermal_algorithms.pipeline import Alert, AlertType, ThermalPipeline
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from examples.utils import find_session, load_frames, make_synthetic_homographies


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIB_DIR    = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"

# Any recording with all three channels works.
SCENE_CANDIDATES = [
    DATASET_ROOT / "lighter_on_3_sec",
    DATASET_ROOT / "lighter_on_and_off_3_sec",
    DATASET_ROOT / "personpresence",
    DATASET_ROOT / "1_man_jump_in_place_4_sec",
]

OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "pipeline_demo.png"


# ---------------------------------------------------------------------------
# Helpers (session-specific to the pipeline demo)
# ---------------------------------------------------------------------------

def _load_three_channel_frames(session_dir: Path) -> list[tuple[Frame, Frame, Frame]] | None:
    """Load synchronised 3-camera frames from a session directory."""
    arrays = []
    for ch in range(3):
        npz = session_dir / f"ch{ch}_raw_data.npz"
        if not npz.is_file():
            return None
        arrays.append(np.load(npz)["frames"].astype(np.float32))

    n_frames = min(a.shape[0] for a in arrays)
    fps = 8.0
    return [
        (
            Frame(data=arrays[0][i], timestamp=i / fps,         camera_id=0),
            Frame(data=arrays[1][i], timestamp=i / fps + 0.02,  camera_id=1),
            Frame(data=arrays[2][i], timestamp=i / fps + 0.04,  camera_id=2),
        )
        for i in range(n_frames)
    ]


def _make_synthetic_triplets(n_frames: int = 60) -> list[tuple[Frame, Frame, Frame]]:
    """Synthetic clip: fire appears at frame 20 and grows; contact at frame 40."""
    rng = np.random.default_rng(7)
    H, W = MLX90640.height, MLX90640.width
    fps = 8.0
    triplets = []
    for i in range(n_frames):
        frames = []
        for cam_id in range(3):
            data = (rng.standard_normal((H, W)) * 0.5 + 25.0).astype(np.float32)
            # Fire blob: appears at frame 20, grows
            if i >= 20:
                size = max(1, (i - 19) // 3)
                r, c = H // 4, W // 4
                data[r - size:r + size + 1, c - size:c + size + 1] = 66.0
            # Two-person blobs (for contact detection visual)
            if i >= 40:
                # Person A — left
                data[H // 2 - 3:H // 2 + 4, W // 3 - 2:W // 3 + 3] = 36.5
                # Person B — close to A
                data[H // 2 - 3:H // 2 + 4, W // 3 + 2:W // 3 + 7] = 36.5
            frames.append(Frame(data=data, timestamp=i / fps + cam_id * 0.02,
                                camera_id=cam_id))
        triplets.append(tuple(frames))  # type: ignore[arg-type]
    return triplets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Load or synthesise frame triplets ────────────────────────────────
    session_dir = find_session(SCENE_CANDIDATES, require_channels=3)
    if session_dir is not None:
        triplets = _load_three_channel_frames(session_dir)
        if triplets is None:
            print(f"Session {session_dir} has fewer than 3 channels — using synthetic data.")
            session_dir = None

    if session_dir is None:
        triplets = _make_synthetic_triplets()
        src_label = "synthetic (fire at t≈2.5s, contact at t≈5.0s)"
        print(f"Using {len(triplets)} synthetic frame triplets.")
    else:
        src_label = f"{session_dir.parent.name}/{session_dir.name}"
        print(f"Loaded {len(triplets)} frame triplets from {src_label}")

    # ── 2. Calibrate preprocessor ───────────────────────────────────────────
    calib_npz = CALIB_DIR / "ch0_raw_data.npz"
    if calib_npz.is_file():
        calib_data = np.load(calib_npz)["frames"].astype(np.float32)
        calib_frames = [Frame(data=calib_data[i], timestamp=i / 8.0, camera_id=0)
                        for i in range(calib_data.shape[0])]
        preprocessor = TatenoPipeline(MLX90640).fit(calib_frames)
        print(f"Tateno fitted on {len(calib_frames)} frames (σ={preprocessor.sigma:.2f}px)")
    else:
        print("No calibration data found — preprocessor omitted (raw mode).")
        preprocessor = None

    # ── 3. Assemble the pipeline ─────────────────────────────────────────────
    H = make_synthetic_homographies()
    pipeline = ThermalPipeline(
        preprocessor=preprocessor,
        human_detector=AdaptiveThresholdDetector(MLX90640).fit([]),
        fire_detector=OtsuFireDetector(
            MLX90640, t_ign=45.0, t_fire=60.0,
            delta_t=1.0, tau_step=4, s_threshold=2, t_measure=8.0,
        ).fit([]),
        contact_detector=GeometricContactDetector(
            homography=H, epsilon_m=0.4, delta_m=0.5,
        ).fit([]),
        fire_cooldown_s=5.0,      # shorter for demo
        contact_cooldown_s=5.0,
    )
    print(f"\nPipeline: {pipeline}")

    # ── 4. Process all triplets ──────────────────────────────────────────────
    results = []
    for f0, f1, f2 in triplets:
        results.append(pipeline.process(f0, f1, f2))

    alert_times  = {"fire": [], "contact": []}
    for r in results:
        for alert in r.alerts:
            if alert.type == AlertType.FIRE:
                alert_times["fire"].append(r.timestamp)
            elif alert.type == AlertType.CONTACT:
                alert_times["contact"].append(r.timestamp)

    print(f"\nAlert summary over {len(results)} frames:")
    print(f"  FIRE alerts:    {len(alert_times['fire'])}   "
          f"at t = {[f'{t:.1f}' for t in alert_times['fire']]}")
    print(f"  CONTACT alerts: {len(alert_times['contact'])}  "
          f"at t = {[f'{t:.1f}' for t in alert_times['contact']]}")

    # ── 5. Pick a representative frame to visualise ──────────────────────────
    # Prefer a fire-alarm frame; fall back to last frame
    alarm_indices = [i for i, r in enumerate(results) if r.fire_alarm]
    vis_idx = alarm_indices[0] if alarm_indices else len(results) // 2
    vis_result = results[vis_idx]
    vis_triplet = triplets[vis_idx]

    # ── 6. Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.1)

    # — Row 0: raw camera frames —
    for cam_id, frame in enumerate(vis_triplet):
        ax = fig.add_subplot(gs[0, cam_id])
        vmin, vmax = np.percentile(frame.data, [2, 98])
        ax.imshow(frame.data, cmap="inferno", vmin=vmin, vmax=vmax)
        for d in vis_result.detections[cam_id]:
            x, y, w, h = d.bbox
            rect = mpatches.Rectangle((x - 0.5, y - 0.5), w, h,
                                      linewidth=1.5, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
        ax.set_title(f"Camera {cam_id} — raw\nt={frame.timestamp:.2f}s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # — Row 1: preprocessed + fire level badges —
    for cam_id, proc_frame in enumerate(vis_result.processed_frames):
        ax = fig.add_subplot(gs[1, cam_id])
        vmin, vmax = np.percentile(proc_frame.data, [2, 98])
        ax.imshow(proc_frame.data, cmap="viridis", vmin=vmin, vmax=vmax)
        fire_lv = vis_result.fire_alerts[cam_id].level
        badge_col = {"safe": "#2ecc71", "potential_fire": "#f39c12",
                     "ignition_source": "#e67e22", "active_combustion": "#e74c3c"}
        ax.set_title(
            f"Camera {cam_id} — preprocessed\nFire: {fire_lv.value.upper()}",
            fontsize=9,
            color=badge_col.get(fire_lv.value, "white"),
        )
        ax.set_xticks([]); ax.set_yticks([])
        if fire_lv.value != "safe":
            for spine in ax.spines.values():
                spine.set_edgecolor(badge_col.get(fire_lv.value, "white"))
                spine.set_linewidth(3)

    # — Row 2 (spanning all columns): alert timeline —
    ax_tl = fig.add_subplot(gs[2, :])
    times = [r.timestamp for r in results]

    # Background: SAFE = dark, with alert stripes
    ax_tl.set_xlim(times[0], times[-1])
    ax_tl.set_ylim(-0.5, 1.5)
    ax_tl.set_facecolor("#1a1a2e")

    # Shade frames by alarm state
    for i, r in enumerate(results[:-1]):
        t_start, t_end = times[i], times[i + 1]
        if r.fire_alarm:
            ax_tl.axvspan(t_start, t_end, ymin=0.5, ymax=1.0, alpha=0.7, color="#e74c3c")
        if r.contact_alarm:
            ax_tl.axvspan(t_start, t_end, ymin=0.0, ymax=0.5, alpha=0.7, color="#9b59b6")

    ax_tl.axhline(0.5, color="gray", linewidth=0.5, alpha=0.5)
    ax_tl.set_yticks([0.25, 0.75])
    ax_tl.set_yticklabels(["CONTACT", "FIRE"], fontsize=9, color="white")
    ax_tl.set_xlabel("Time (s)", color="white")
    ax_tl.tick_params(colors="white")
    for spine in ax_tl.spines.values():
        spine.set_edgecolor("gray")

    # Mark the visualised frame
    ax_tl.axvline(vis_result.timestamp, color="white", linewidth=1.5,
                  linestyle="--", alpha=0.8, label="visualised frame ↑")
    ax_tl.legend(loc="upper right", fontsize=8, labelcolor="white",
                 facecolor="#111", edgecolor="gray")

    legend_patches = [
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="FIRE alert"),
        mpatches.Patch(color="#9b59b6", alpha=0.7, label="CONTACT alert"),
        mpatches.Patch(color="#1a1a2e", label="SAFE (no alarm)"),
    ]
    ax_tl.legend(handles=legend_patches, loc="upper right", fontsize=8,
                 labelcolor="white", facecolor="#111", edgecolor="gray")

    fig.suptitle(
        f"ThermalPipeline — end-to-end demo (Figure 5 flow)\n"
        f"Source: {src_label}  |  "
        f"FIRE alerts: {len(alert_times['fire'])}  |  "
        f"CONTACT alerts: {len(alert_times['contact'])}",
        fontsize=11, color="white",
    )
    fig.patch.set_facecolor("#111111")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="#111111")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
