"""Render 3-camera thermal strips for suspect segments so labels can be verified by eye."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import eval_waveshare_contact as geo
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "outputs" / "label_audit"
OUT.mkdir(parents=True, exist_ok=True)

# (scene, frames_to_render, note)
TARGETS = [
    ("the_more_the_merrier", [117, 141, 150, 163, 172, 188, 224, 235, 249], "labeled 0, predicted 1"),
    ("3ppl_dance",           [8, 11, 94, 97, 101], "labeled 0, predicted 1"),
    ("3pp_surprise",         [0, 10, 20, 30, 38, 57, 61, 65], "labeled 1, predicted 0"),
    ("2ppl_hug",             [55, 60, 62, 64, 69, 72], "labeled 1, predicted 0 (60-72)"),
    ("2ppl_fight",           [58, 65, 74, 82], "labeled 0, predicted 1"),
    ("3pplhedroncolider",    [54, 60, 70, 79, 193, 197, 201], "labeled 0, predicted 1"),
]

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
labels = geo._load_contact_labels(geo.CONTACT_CSV)

for scene, fis, note in TARGETS:
    s = idx.find(scene)
    fis = [fi for fi in fis if fi < s.n_frames]
    fig, axes = plt.subplots(len(fis), len(geo.CHANNELS),
                             figsize=(3.2 * len(geo.CHANNELS), 2.6 * len(fis)))
    axes = np.atleast_2d(axes)
    for row, fi in enumerate(fis):
        lab = labels[scene].get(fi, "?")
        for col, ch in enumerate(geo.CHANNELS):
            fr = geo.get_frame(s, ch, fi)
            ax = axes[row, col]
            ax.imshow(fr.data, cmap="inferno")
            for (x, y, w, h) in [d.bbox for d in ssd.predict(fr)]:
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="cyan", lw=1.2))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"f{fi} lab={lab}", fontsize=9)
            if row == 0:
                ax.set_title(ch, fontsize=9)
    fig.suptitle(f"{scene} — {note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / f"{scene.replace(' ', '_')}.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    print("wrote", p)
