"""
Option 4 — Combined 2×2 Panel
Top-left:  Fire detection bar chart (Prec / Rec / F1)
Top-right: Human detection F1 under Crit-A vs Crit-B (grouped bar)
Bottom-left:  Confusion-matrix heat tiles for fire detectors
Bottom-right: Human localisation gap delta (horizontal bar)
Run: python scripts/plot_option4_combined_panel.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Smart Thermal System — Detection Algorithm Results\n"
             "MLX90640 · 32×24 px · 22 Scenes",
             fontsize=13, fontweight="bold", y=0.99)

# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEFT: Fire bar chart
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0, 0]

det_names  = ["Otsu\nOriginal", "Otsu\nBypass", "Fire\nSVM"]
precision  = [70.0, 40.4, 100.0]
recall     = [ 9.0, 26.9,  32.0]
f1         = [15.9, 32.3,  48.5]

x     = np.arange(3)
w     = 0.24
b1 = ax.bar(x - w, precision, w, color="#2196F3", alpha=0.85, label="Precision", zorder=3)
b2 = ax.bar(x,     recall,    w, color="#F44336", alpha=0.85, label="Recall",    zorder=3)
b3 = ax.bar(x + w, f1,        w, color="#4CAF50", alpha=0.85, label="F1",        zorder=3)

for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                f"{h:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.axhline(27, color="#FF9800", linewidth=1.2, linestyle="--", zorder=2)
ax.text(2.55, 29, "27% ceiling", color="#E65100", fontsize=7.5, ha="right")

ax.set_xticks(x)
ax.set_xticklabels(det_names, fontsize=9)
ax.set_ylim(0, 118)
ax.set_ylabel("Score (%)")
ax.set_title("Fire Detection", fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")

# ─────────────────────────────────────────────────────────────────────────────
# TOP-RIGHT: Human F1 grouped bar (Crit A vs B)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0, 1]

algos   = ["Adaptive\nThreshold", "HOG+SVM", "HOG+SVM\n+Pyramid", "MobileNet\nSSD"]
f1_a    = [98.5, 97.8, None, 98.8]
f1_b    = [82.0, 57.4, 75.5, 94.9]
colors_a = ["#A5D6A7", "#80CBC4", "#FFE082", "#90CAF9"]
colors_b = ["#2E7D32", "#00695C", "#F57F17", "#1565C0"]

x = np.arange(len(algos))
w = 0.36

for i, (fa, fb, ca, cb) in enumerate(zip(f1_a, f1_b, colors_a, colors_b)):
    if fa is not None:
        b = ax.bar(i - w/2, fa, w, color=ca, alpha=0.9, zorder=3)
        ax.text(i - w/2, fa + 0.7, f"{fa:.1f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333")
    b2 = ax.bar(i + w/2, fb, w, color=cb, alpha=0.9, zorder=3)
    ax.text(i + w/2, fb + 0.7, f"{fb:.1f}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color=cb)

import matplotlib.patches as mpatches
pa = mpatches.Patch(color="#888888", alpha=0.35, label="Crit A — presence only")
pb = mpatches.Patch(color="#888888", alpha=0.9,  label="Crit B — IoU > 0.5")
ax.legend(handles=[pa, pb], frameon=False, fontsize=8.5, loc="lower left")

ax.set_xticks(x)
ax.set_xticklabels(algos, fontsize=9)
ax.set_ylim(40, 106)
ax.set_ylabel("F1-Score (%)")
ax.set_title("Human Detection — F1 by Criterion", fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM-LEFT: Confusion-matrix tile comparison (fire, test split)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1, 0]
ax.axis("off")

# Show 2×2 confusion cells for Otsu Bypass and Fire SVM side by side
cfgs = [
    ("Otsu Hot-Pixel Bypass",  5,  20, 3, 1787),
    ("Fire SVM (RBF)",         8,  17, 0, 1790),
]

cell_colors = {
    "TP": "#C8EFC8", "FN": "#FFF59D",
    "FP": "#FFCDD2", "TN": "#BBDEFB",
}

for col_idx, (name, tp, fn, fp, tn) in enumerate(cfgs):
    base_x = 0.05 + col_idx * 0.50
    base_y = 0.55
    cw, ch = 0.20, 0.18

    ax.text(base_x + cw, base_y + 0.22, name, ha="center", va="bottom",
            fontsize=9, fontweight="bold", transform=ax.transAxes)

    for row, (rlabel, cv_left, cv_right) in enumerate(
            [("Truth: Fire", ("TP", tp), ("FN", fn)),
             ("Truth: Safe", ("FP", fp), ("TN", tn))]):

        for col, (cname, val) in enumerate([cv_left, cv_right]):
            rect = patches.FancyBboxPatch(
                (base_x + col * cw, base_y - row * ch),
                cw - 0.01, ch - 0.01,
                boxstyle="round,pad=0.005",
                facecolor=cell_colors[cname],
                edgecolor="#aaaaaa", linewidth=0.8,
                transform=ax.transAxes, zorder=3)
            ax.add_patch(rect)
            ax.text(base_x + col * cw + cw / 2,
                    base_y - row * ch + ch / 2,
                    f"{cname}\n{val}",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    transform=ax.transAxes)

    # column headers
    for col, label in enumerate(["Pred: Fire", "Pred: Safe"]):
        ax.text(base_x + col * cw + cw / 2, base_y + ch * 0.55, label,
                ha="center", va="center", fontsize=8,
                transform=ax.transAxes, style="italic")

ax.set_title("Fire Detection — Confusion Matrices (test split, 1 815 frames)",
             fontweight="bold")

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM-RIGHT: Localisation gap (horizontal bar)
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1, 1]

algos_h = ["MobileNet-SSD", "Adaptive Thresh.", "HOG+SVM+Pyramid", "HOG+SVM"]
f1_bvals = [94.9, 82.0, 75.5, 57.4]
colors_h  = ["#1565C0", "#2E7D32", "#F57F17", "#C62828"]

y = np.arange(len(algos_h))
bars = ax.barh(y, f1_bvals, color=colors_h, alpha=0.85, zorder=3, height=0.55)

for bar, val in zip(bars, f1_bvals):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(algos_h, fontsize=9.5)
ax.set_xlim(40, 105)
ax.set_xlabel("F1-Score (IoU > 0.5, %)")
ax.set_title("Human Detection — Localisation F1\n(Criterion B, test set)", fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
ax.invert_yaxis()

# ─────────────────────────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("scripts/output_option4_combined_panel.png", dpi=150, bbox_inches="tight")
print("Saved: scripts/output_option4_combined_panel.png")
plt.show()
