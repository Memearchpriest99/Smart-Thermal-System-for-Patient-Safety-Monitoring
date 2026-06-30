"""Visualize config-D failures on 3pplhedroncolider.

Renders:
  (1) outputs/hedron_timeline.png  — per-frame outcome (TP/FN/FP/TN) across the
      whole scene, plus people-count and #cameras-firing traces, so you can see
      WHERE the errors fall.
  (2) outputs/hedron_FN.png        — every missed-contact frame, 3 cameras each,
      with SSD person boxes (lime) and the warm-blob segmentation (cyan tint)
      so you can see WHY the two-body-merge test did not fire.
  (3) outputs/hedron_FP.png        — a sample of false-alarm frames, same overlay.

config-D = raw SSD + residual blob + two-body merge (k=1, tau2=8) + T1 morph (2,7).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label as cc_label

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_v8 as v8
import eval_waveshare_contact_v9 as v9
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

SCENE = "3pplhedroncolider"
K, TAU2 = 1.0, 8.0
LO, LC = 2, 7
H, W = geo.PROFILE.height, geo.PROFILE.width
OUT = _ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def cam_detail(boxes, resid):
    """cleaned boxes, warm mask, n, and the EXACT config-D per-camera state."""
    cleaned = vv.merge_oversegmented([b for b in boxes])
    thr = float(resid.mean() + K * resid.std())
    mask = resid > thr
    state = v9._cam_state_2body(cleaned, resid, K, TAU2)   # 'T'/'N'/'M'/'C'
    return cleaned, mask, len(cleaned), state


def render(ax, resid, boxes, mask, title, border):
    vmin, vmax = np.percentile(resid, [5, 99])
    ax.imshow(resid, cmap="inferno", vmin=vmin, vmax=vmax)
    ax.imshow(np.ma.masked_where(~mask, mask), cmap="cool", alpha=0.30)
    for b in boxes:
        ax.add_patch(mpatches.Rectangle((b[0] - .5, b[1] - .5), b[2], b[3],
                     lw=1.4, edgecolor="lime", facecolor="none"))
        ax.plot(b[0] + b[2] / 2, b[1] + b[3] / 2, "o", ms=3, color="white")
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(border); sp.set_linewidth(2.5)


def main():
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
    ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    s = idx.find(SCENE)
    fis = sorted(fi for fi in labels[SCENE] if 0 <= fi < s.n_frames)
    print(f"{SCENE}: {len(fis)} frames", flush=True)

    rows = []
    for fi in fis:
        cams = []
        for ch in (0, 1, 2):
            rawF = geo.get_frame(s, ch, fi)
            resid = pre[ch].predict(rawF).data.astype(np.float32)
            boxes = [d.bbox for d in ssd.predict(rawF)]
            cleaned, mask, n, state = cam_detail(boxes, resid)
            cams.append(dict(resid=resid, boxes=cleaned, mask=mask, n=n, state=state,
                             touch=(state == "T")))
        people = max(c["n"] for c in cams)
        states = [c["state"] for c in cams]
        votes = states.count("T"); merged = states.count("M"); clear = "C" in states
        # config-D V9-core decision (no X3D): people<=1 ->0 else two-body quorum
        base = 0 if people <= 1 else (1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0)
        rows.append(dict(fi=fi, label=int(labels[SCENE][fi]), base=base,
                         people=people, votes=votes, cams=cams))

    # temporal morphology over base
    sm = v8.morph([r["base"] for r in rows], LO, LC)
    for r, p in zip(rows, sm):
        r["pred"] = p
        r["outcome"] = ("TP" if r["label"] and p else "FN" if r["label"] else
                        "FP" if p else "TN")

    fn = [r for r in rows if r["outcome"] == "FN"]
    fp = [r for r in rows if r["outcome"] == "FP"]
    tp = [r for r in rows if r["outcome"] == "TP"]
    print(f"TP={len(tp)} FN={len(fn)} FP={len(fp)}")
    print("FN frames:", [r["fi"] for r in fn])
    print("FP frames:", [r["fi"] for r in fp])

    # ---- (1) timeline ----
    color = {"TP": "#2ecc71", "FN": "#e74c3c", "FP": "#e67e22", "TN": "#34495e"}
    fig, axs = plt.subplots(3, 1, figsize=(15, 6), height_ratios=[1.1, 1, 1], sharex=True)
    xs = list(range(len(rows)))
    axs[0].bar(xs, [1] * len(rows), width=1.0,
               color=[color[r["outcome"]] for r in rows])
    axs[0].set_yticks([]); axs[0].set_ylabel("outcome")
    axs[0].set_title(f"{SCENE} — config-D outcome per frame  "
                     f"(TP={len(tp)} FN={len(fn)} FP={len(fp)})", fontsize=11)
    handles = [mpatches.Patch(color=color[k], label=k) for k in ("TP", "FN", "FP", "TN")]
    axs[0].legend(handles=handles, ncol=4, loc="upper right", fontsize=8)
    axs[1].plot(xs, [r["label"] for r in rows], color="black", lw=1.2, label="GT contact")
    axs[1].plot(xs, [r["pred"] for r in rows], color="#e67e22", lw=1.0, alpha=0.8, label="config-D")
    axs[1].set_yticks([0, 1]); axs[1].set_ylabel("contact"); axs[1].legend(fontsize=8, loc="upper right")
    axs[2].plot(xs, [r["people"] for r in rows], color="#2980b9", label="people detected (max cam)")
    axs[2].plot(xs, [r["votes"] for r in rows], color="#8e44ad", label="#cams firing (2-body)")
    axs[2].set_ylabel("count"); axs[2].set_xlabel("frame (index within scene)")
    axs[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "hedron_timeline.png", dpi=130); plt.close(fig)

    # ---- (2) FN gallery ---- and ---- (3) FP gallery ----
    def gallery(items, name, kind):
        if not items:
            return
        nr = len(items)
        fig, axs = plt.subplots(nr, 3, figsize=(7.5, 2.3 * nr), squeeze=False)
        for ri, r in enumerate(items):
            stname = {"T": "2-BODY MERGE", "N": "near, no merge", "M": "1 blob (over-merged)",
                      "C": "separated/none"}
            for ch in range(3):
                c = r["cams"][ch]
                render(axs[ri][ch], c["resid"], c["boxes"], c["mask"],
                       f"f{r['fi']} cam{ch}: {c['n']}box {stname[c['state']]}",
                       border=("#e74c3c" if kind == "FN" else "#e67e22"))
            axs[ri][0].set_ylabel(f"frame {r['fi']}", fontsize=8)
        fig.suptitle(f"{SCENE} — {name}  (lime=person box, cyan=warm blob; "
                     f"contact fires only when 2 boxes share one warm blob)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(OUT / f"hedron_{kind}.png", dpi=120); plt.close(fig)

    gallery(fn, "MISSED contacts (FN)", "FN")
    gallery(fp[:: max(1, len(fp) // 10)][:10], "FALSE alarms (FP, sampled)", "FP")
    print("saved: outputs/hedron_timeline.png, hedron_FN.png, hedron_FP.png")


if __name__ == "__main__":
    main()
