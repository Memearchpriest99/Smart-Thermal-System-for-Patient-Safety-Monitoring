"""
Option 1 — Fire Detection: Grouped Bar Chart
Compares Precision / Recall / F1 across the three fire detector configurations.
Run: python scripts/plot_option1_fire_bar.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
detectors = [
    "Otsu\nPipelined\n(original)",
    "Otsu\nHot-Pixel\nBypass",
    "Fire SVM\n(RBF)",
]

precision = [70.0, 40.4, 100.0]
recall    = [ 9.0, 26.9,  32.0]
f1        = [15.9, 32.3,  48.5]

# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "precision": "#2196F3",   # blue
    "recall":    "#F44336",   # red
    "f1":        "#4CAF50",   # green
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

x     = np.arange(len(detectors))
width = 0.24

b1 = ax.bar(x - width, precision, width, label="Precision",
            color=COLORS["precision"], alpha=0.88, zorder=3)
b2 = ax.bar(x,          recall,   width, label="Recall",
            color=COLORS["recall"],    alpha=0.88, zorder=3)
b3 = ax.bar(x + width,  f1,       width, label="F1-Score",
            color=COLORS["f1"],        alpha=0.88, zorder=3)

# Value labels
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold")

# Recall ceiling annotation
ax.axhline(27, color="#FF9800", linewidth=1.4, linestyle="--", zorder=2)
ax.text(2.62, 28.5, "≈27 % recall ceiling\n(label/physics limit)",
        color="#E65100", fontsize=8.5, va="bottom", ha="right")

ax.set_xticks(x)
ax.set_xticklabels(detectors, fontsize=10.5)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_title("Fire Detection — Algorithm Comparison\n"
             "(test split: 1 815 frames, 25 positive)",
             fontsize=12.5, fontweight="bold", pad=12)
ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
ax.legend(frameon=False, fontsize=10.5, loc="upper left")

# Note about eval sets
ax.text(0.01, 0.01,
        "* Otsu configs evaluated on full dataset (5 943 frames).\n"
        "  Fire SVM and Otsu Bypass (head-to-head) on 70/30 test split.",
        transform=ax.transAxes, fontsize=8, color="#666666", va="bottom")

plt.tight_layout()
plt.savefig("scripts/output_option1_fire_bar.png", dpi=150, bbox_inches="tight")
print("Saved: scripts/output_option1_fire_bar.png")
plt.show()
