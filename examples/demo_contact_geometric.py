"""Visual demo: GeometricContactDetector with multi-view homography (§ 4.4.3.1).

Shows the full classical contact-detection pipeline:

    3 camera frames
      └─► HumanDetector (per camera)  →  bounding boxes
      └─► Foot-point extraction        →  P_foot = (x + w/2, y + h)
      └─► Homographic projection       →  floor-plane (X_w, Y_w) per camera
      └─► Cross-camera validation      →  discard single-source, outlier removal
      └─► Clustering → ActorPosition   →  unique person locations on floor map
      └─► Pairwise distance check      →  D_{i,j} < δ  →  CONTACT alert

Visualizes:
  • Top row   — three camera views with detected bounding boxes
  • Bottom    — top-down floor map: projected foot-points per camera + actor
                positions + contact threshold circle, coloured by outcome

Uses a synthetic homography calibrated to a 4 m × 3 m room, so the demo
works without physical Hot-Point Calibration markers.  Replace the homographies
with output from ``detector.calibrate_homography(marker_correspondences)`` for
a real deployment.

Run from the project root:
    python examples/demo_contact_geometric.py

Output: outputs/contact_geometric_demo.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import Detection, Frame, HomographyMatrices
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from examples.utils import make_synthetic_homographies, make_synthetic_triplet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("/sessions/magical-youthful-euler/mnt/dataset")
CALIB_DIR    = DATASET_ROOT / "emptyroomwithpcscreen" / "20260405_195559"

# Use a 2-person recording if available; falls back to synthetic.
SCENE_CANDIDATES = [
    DATASET_ROOT / "2men" ,
    DATASET_ROOT / "2_men_far_near_far",
    DATASET_ROOT / "touch",
]
CONTACT_DISTANCE_M = 0.5   # δ — proximity threshold (metres)
OUTPUT_PATH = Path(__file__).parent.parent / "outputs" / "contact_geometric_demo.png"

# Room dimensions for the synthetic homography (metres)
ROOM_W_M = 4.0
ROOM_H_M = 3.0




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Build synthetic homography (or calibrate from markers) ───────────
    homographies = make_synthetic_homographies(room_w_m=ROOM_W_M, room_h_m=ROOM_H_M)
    print("Homography matrices (synthetic — replace with calibrated H for deployment):")
    for i, H in enumerate([homographies.h1, homographies.h2, homographies.h3]):
        print(f"  H{i+1}[0,:] = {H[0].round(4)}")

    # ── 2. Build the detector ───────────────────────────────────────────────
    detector = GeometricContactDetector(
        homography=homographies,
        epsilon_m=0.4,
        delta_m=CONTACT_DISTANCE_M,
    ).fit([])
    print(f"\nGeometricContactDetector: ε={detector._epsilon_m}m, δ={detector._delta_m}m")

    # ── 3. Prepare two scenarios: no-contact and contact ────────────────────
    #    Positions are (cx_norm, cy_norm) in the world frame [0,1]
    scenarios = [
        {
            "label": "No contact\n(persons 1.5 m apart)",
            "positions": [(0.25, 0.5), (0.75, 0.5)],
        },
        {
            "label": "Contact\n(persons 0.3 m apart)",
            "positions": [(0.45, 0.5), (0.55, 0.5)],  # 0.1 * 4m = 0.4m apart
        },
    ]

    fig, axes = plt.subplots(len(scenarios), 4, figsize=(14, 5.5 * len(scenarios)))
    if len(scenarios) == 1:
        axes = axes[np.newaxis]

    for row_idx, scenario in enumerate(scenarios):
        frames_tuple, gt_dets = make_synthetic_triplet(scenario["positions"])
        frames = list(frames_tuple)
        result = detector.predict(tuple(frames), detections=tuple(gt_dets))
        contact = result.any_contact

        print(f"\nScenario: {scenario['label'].replace(chr(10), ' ')}")
        print(f"  Actors found: {len(result.actors)}")
        for actor in result.actors:
            print(f"    actor track_id={actor.track_id}  "
                  f"world=({actor.world_xy[0]:.2f}, {actor.world_xy[1]:.2f}) m  "
                  f"cameras={actor.source_camera_ids}")
        if result.any_contact:
            print(f"  *** CONTACT DETECTED ***  pairs={result.pairs_in_contact}")
        else:
            print("  No contact detected.")

        # — Camera views (3 panels) —
        for cam_idx in range(3):
            ax = axes[row_idx, cam_idx]
            f = frames[cam_idx]
            vmin, vmax = np.percentile(f.data, [2, 98])
            ax.imshow(f.data, cmap="inferno", vmin=vmin, vmax=vmax)

            for d in gt_dets[cam_idx]:
                x, y, w, h = d.bbox
                rect = mpatches.Rectangle(
                    (x - 0.5, y - 0.5), w, h,
                    linewidth=2,
                    edgecolor="lime" if not contact else "red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                # Foot point
                fp_x = x + w / 2; fp_y = y + h
                ax.plot(fp_x, fp_y, "o", color="cyan", markersize=4)

            ax.set_title(f"Camera {cam_idx}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        # — Floor map —
        ax_floor = axes[row_idx, 3]
        ax_floor.set_xlim(0, ROOM_W_M)
        ax_floor.set_ylim(0, ROOM_H_M)
        ax_floor.set_facecolor("#111")
        ax_floor.set_aspect("equal")
        ax_floor.set_title("Floor map (world coords)", fontsize=9)
        ax_floor.set_xlabel("X (m)"); ax_floor.set_ylabel("Y (m)")

        # Draw room boundary
        room_rect = mpatches.Rectangle((0, 0), ROOM_W_M, ROOM_H_M,
                                       linewidth=1.5, edgecolor="gray", facecolor="none")
        ax_floor.add_patch(room_rect)

        # Plot unique actor positions + contact circle
        for i, actor in enumerate(result.actors):
            ax_floor.plot(*actor.world_xy, "o",
                          color="red" if contact else "lime",
                          markersize=10, zorder=5)
            ax_floor.annotate(
                f"A{i}", actor.world_xy,
                textcoords="offset points", xytext=(4, 4),
                color="white", fontsize=8,
            )

        # Draw distance between actors if exactly 2
        if len(result.actors) == 2:
            ax_floor.plot(
                [result.actors[0].world_xy[0], result.actors[1].world_xy[0]],
                [result.actors[0].world_xy[1], result.actors[1].world_xy[1]],
                "--", color="red" if contact else "yellow", linewidth=1.2, alpha=0.7,
            )
            mid = (
                (result.actors[0].world_xy[0] + result.actors[1].world_xy[0]) / 2,
                (result.actors[0].world_xy[1] + result.actors[1].world_xy[1]) / 2,
            )
            from math import sqrt
            dist = sqrt(sum((a - b) ** 2
                           for a, b in zip(result.actors[0].world_xy,
                                           result.actors[1].world_xy)))
            ax_floor.text(mid[0], mid[1] + 0.1, f"d={dist:.2f}m",
                         color="white", fontsize=8, ha="center")

        # Draw δ threshold circles around each actor
        for actor in result.actors:
            circle = plt.Circle(actor.world_xy, CONTACT_DISTANCE_M,
                                color="orange", fill=False, linestyle=":",
                                linewidth=1, alpha=0.5)
            ax_floor.add_patch(circle)

        # Outcome label
        outcome_text = "⚠ CONTACT" if contact else "✓ SAFE"
        outcome_col  = "red" if contact else "lime"
        ax_floor.set_title(
            f"Floor map — {outcome_text}",
            color=outcome_col, fontsize=9, fontweight="bold",
        )

        # Scenario row label
        axes[row_idx, 0].set_ylabel(
            scenario["label"], fontsize=9, rotation=0,
            ha="right", va="center", labelpad=60,
        )

    fig.suptitle(
        "GeometricContactDetector (§ 4.4.3.1)\n"
        "Homographic foot-point projection → cross-camera fusion → "
        f"proximity check (δ = {CONTACT_DISTANCE_M} m)",
        fontsize=11,
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=140, bbox_inches="tight", facecolor="#222")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
