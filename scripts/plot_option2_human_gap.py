"""
Option 2 — Human Detection: Localisation Gap (Slope Chart)
Shows how each algorithm's F1 drops when IoU > 0.5 is required.
The steeper the line, the worse the localisation.
Run: python scripts/plot_option2_human_gap.py
"""

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
# (algorithm, F1_noIoU, F1_IoU, color)
algorithms = [
    ("MobileNet-SSD",       98.8, 94.9, "#1565C0"),
    ("Adaptive Threshold",  98.5, 82.0, "#2E7D32"),
    ("HOG+SVM+Pyramid",     None, 75.5, "#F57F17"),   # no Crit-A reported
    ("HOG+SVM",             97.8, 57.4, "#C62828"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})

fig, ax = plt.subplots(figsize=(8, 6))

col_a, col_b = 0, 1
ax.set_xlim(-0.35, 1.35)
ax.set_ylim(40, 105)

# Column headers
for x_pos, label in [(col_a, "Criterion A\n(Presence only)"),
                     (col_b, "Criterion B\n(IoU > 0.5)")]:
    ax.text(x_pos, 103, label, ha="center", va="top",
            fontsize=11, fontweight="bold", color="#333333")

ax.axvline(col_a, color="#cccccc", linewidth=1, zorder=0)
ax.axvline(col_b, color="#cccccc", linewidth=1, zorder=0)

for algo, f1_a, f1_b, color in algorithms:
    if f1_a is not None:
        # draw slope line
        ax.plot([col_a, col_b], [f1_a, f1_b],
                color=color, linewidth=2.0, alpha=0.85, zorder=3)
        # left dot + label
        ax.scatter(col_a, f1_a, color=color, s=70, zorder=4)
        ax.text(col_a - 0.04, f1_a, f"{f1_a:.1f}%",
                ha="right", va="center", fontsize=9.5,
                fontweight="bold", color=color)
    else:
        # dashed from implied position for pyramid (same SVM weights, no Crit-A row)
        ax.scatter(col_b, f1_b, color=color, s=70, zorder=4, marker="D")

    # right dot + label
    ax.scatter(col_b, f1_b, color=color, s=70, zorder=4)
    ax.text(col_b + 0.04, f1_b, f"{f1_b:.1f}%",
            ha="left", va="center", fontsize=9.5,
            fontweight="bold", color=color)

    # drop annotation (only for slopes)
    if f1_a is not None:
        drop = f1_a - f1_b
        mid_x = 0.5
        mid_y = (f1_a + f1_b) / 2
        ax.text(mid_x, mid_y, f"−{drop:.1f} pp",
                ha="center", va="center", fontsize=8.5,
                color=color, bbox=dict(facecolor="white", edgecolor="none",
                                       alpha=0.8, pad=1))

# Legend
handles = [mlines.Line2D([], [], color=c, linewidth=2, label=n)
           for n, *_, c in algorithms]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Human Detection — Localisation Gap\n"
             "F1 with vs. without IoU > 0.5 requirement  (test set, 621 frames)",
             fontsize=12.5, fontweight="bold", pad=14)

plt.tight_layout()
plt.savefig("scripts/output_option2_human_gap.png", dpi=150, bbox_inches="tight")
print("Saved: scripts/output_option2_human_gap.png")
plt.show()
