"""
Option 3 — Precision-Recall Scatter with F1 Iso-curves
All fire and human-detection configurations on one canvas.
Iso-curves show constant F1 = {0.50, 0.75, 0.90}.
Run: python scripts/plot_option3_pr_scatter.py
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
# (label, precision, recall, marker, color, group)
points = [
    # Fire detection
    ("Otsu Original",       70.0,  9.0, "o", "#EF5350", "Fire"),
    ("Otsu Bypass",         40.4, 26.9, "o", "#FF7043", "Fire"),
    ("Fire SVM",           100.0, 32.0, "o", "#B71C1C", "Fire"),

    # Human — Criterion A (presence)
    ("Adaptive (A)",        99.8, 97.2, "s", "#66BB6A", "Human Crit A"),
    ("HOG+SVM (A)",         99.5, 96.2, "s", "#26A69A", "Human Crit A"),
    ("MobileNet (A)",      100.0, 97.5, "s", "#1565C0", "Human Crit A"),

    # Human — Criterion B (IoU > 0.5)
    ("Adaptive (B)",        99.8, 69.6, "^", "#388E3C", "Human Crit B"),
    ("HOG+SVM (B)",         98.8, 40.5, "^", "#00796B", "Human Crit B"),
    ("HOG+SVM+Pyr (B)",     96.9, 61.8, "^", "#0288D1", "Human Crit B"),
    ("MobileNet (B)",      100.0, 90.3, "^", "#1A237E", "Human Crit B"),
]

# ── F1 iso-curves ─────────────────────────────────────────────────────────────
def f1_recall_from_prec(p_arr, f1):
    """Given precision array and target F1, solve for recall."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (f1 * p_arr) / (2 * p_arr - f1)
    r[(r <= 0) | (r > 100)] = np.nan
    return r

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(9, 7))

# draw iso-curves
p_range = np.linspace(0.1, 100, 500)
for f1_target, label in [(90, "F1=90%"), (75, "F1=75%"), (50, "F1=50%")]:
    r_iso = f1_recall_from_prec(p_range, f1_target)
    ax.plot(r_iso, p_range, "--", color="#aaaaaa", linewidth=0.9, zorder=1)
    # find a nice spot to place the label (where both axes are visible)
    valid = np.where(np.isfinite(r_iso) & (r_iso > 5) & (r_iso < 95))[0]
    if len(valid):
        idx = valid[len(valid) // 2]
        ax.text(r_iso[idx] - 1.5, p_range[idx] + 1.2, label,
                fontsize=8, color="#999999", rotation=-28, ha="right")

# group aesthetics
group_alpha = {"Fire": 0.92, "Human Crit A": 0.85, "Human Crit B": 0.85}

for label, prec, rec, marker, color, group in points:
    ax.scatter(rec, prec, marker=marker, color=color, s=100,
               alpha=group_alpha[group], zorder=4,
               edgecolors="white", linewidths=0.6)
    # offset label to avoid overlap
    ax.annotate(label, (rec, prec),
                xytext=(5, 4), textcoords="offset points",
                fontsize=8.5, color=color,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# legend for shapes
import matplotlib.lines as mlines
leg = [
    mlines.Line2D([], [], marker="o", color="#888888", linestyle="None",
                  markersize=8, label="Fire detection"),
    mlines.Line2D([], [], marker="s", color="#888888", linestyle="None",
                  markersize=8, label="Human — Crit A (presence)"),
    mlines.Line2D([], [], marker="^", color="#888888", linestyle="None",
                  markersize=8, label="Human — Crit B (IoU > 0.5)"),
]
ax.legend(handles=leg, frameon=False, fontsize=10, loc="lower left")

ax.set_xlim(-2, 105)
ax.set_ylim(20, 106)
ax.set_xlabel("Recall (%)", fontsize=11)
ax.set_ylabel("Precision (%)", fontsize=11)
ax.set_title("Precision–Recall Space — All Algorithms\n"
             "(dashed curves = constant F1)",
             fontsize=12.5, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.35, zorder=0)

plt.tight_layout()
plt.savefig("scripts/output_option3_pr_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: scripts/output_option3_pr_scatter.png")
plt.show()
