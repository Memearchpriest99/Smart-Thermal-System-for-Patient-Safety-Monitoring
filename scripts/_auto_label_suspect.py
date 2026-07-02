"""Consensus auto-labelling of the 3 mislabelled videos.

config-D (rule-based, no label leak) and the CLEAN-trained Thermo-X3D (never saw
these videos -> no leak) both predict every annotated frame of the 3 suspect
videos. Agreement -> proposed label. Disagreement -> rendered for Guy's visual
confirmation (3-camera strip with raw-SSD boxes).

Outputs:
  reports/auto_label_proposals.json          per-frame preds + proposals
  outputs/label_audit/review_<scene>/f*.png  one strip per disagreement frame
"""
import json, sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.training import DatasetIndex

SUSPECT = ["3pp_surprise", "3ppl_dance", "the_more_the_merrier"]
X3D_CKPT = _ROOT / "checkpoints" / "thermo_x3d_detector" / "Waveshare_26984_clean.thalg"
X3D_TH = 0.1          # val-tuned threshold from the clean retrain
LO, LC = 2, 7
OUT_DIR = _ROOT / "outputs" / "label_audit"

idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
labels = geo._load_contact_labels(geo.CONTACT_CSV)
pre = geo.fit_preprocessors(idx)
ssd_raw = MobileNetSSDDetector.load(ab.RAW_CKPT)
x3d = ThermoX3DDetector.load(X3D_CKPT)

result = {}
t0 = time.time(); n = 0
for scene in SUSPECT:
    s = idx.find(scene)
    fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
    x3d.reset()
    recs = []
    for fi in fis:
        raws = [geo.get_frame(s, ch, fi) for ch in geo.CHANNELS]
        resids = tuple(pre[ch].predict(raws[c]) for c, ch in enumerate(geo.CHANNELS))
        dets_r = [ssd_raw.predict(rw) for rw in raws]
        d_base = ab.v9core_decision(
            {"cams": [{"boxes": [d.bbox for d in dets_r[c]],
                       "resid": resids[c].data.astype(np.float32)} for c in range(3)]}, "resid")
        ev = x3d.predict(resids)
        recs.append({"fi": fi, "old_label": int(labels[scene][fi]),
                     "d_base": int(d_base),
                     "x3d_conf": float(ev.confidence),
                     "x3d_warmup": ev.debug.get("status") == "buffer_filling",
                     "boxes": [[list(map(float, d.bbox)) for d in cam] for cam in dets_r]})
        n += 1
        if n % 150 == 0:
            print(f"   ... {n} frames ({(time.time()-t0)/n*1000:.0f} ms/f)", flush=True)
    # config-D morphology over the whole scene stream
    sm = v8.morph([r["d_base"] for r in recs], LO, LC)
    for r, p in zip(recs, sm):
        r["configD"] = int(p)
        r["x3d"] = int(r["x3d_conf"] > X3D_TH)
        r["agree"] = r["configD"] == r["x3d"]
        r["proposed"] = r["configD"] if r["agree"] else None
    result[scene] = recs

# ---- summary -------------------------------------------------------------
print("\n" + "=" * 78)
print(f"  {'scene':<24} {'N':>4} | {'agree=1':>7} {'agree=0':>7} {'disagree':>8} | {'D-only':>6} {'X3D-only':>8}")
print("  " + "-" * 74)
for scene, recs in result.items():
    a1 = sum(1 for r in recs if r["agree"] and r["configD"] == 1)
    a0 = sum(1 for r in recs if r["agree"] and r["configD"] == 0)
    dis = [r for r in recs if not r["agree"]]
    donly = sum(1 for r in dis if r["configD"] == 1)
    xonly = len(dis) - donly
    print(f"  {scene:<24} {len(recs):>4} | {a1:>7} {a0:>7} {len(dis):>8} | {donly:>6} {xonly:>8}")

# ---- render disagreement frames -------------------------------------------
print("\nrendering disagreement strips ...")
for scene, recs in result.items():
    s = idx.find(scene)
    out = OUT_DIR / f"review_{scene}"
    out.mkdir(parents=True, exist_ok=True)
    for r in recs:
        if r["agree"]:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
        for c, ch in enumerate(geo.CHANNELS):
            fr = geo.get_frame(s, ch, r["fi"])
            axes[c].imshow(fr.data, cmap="inferno")
            for (x, y, w, h) in r["boxes"][c]:
                axes[c].add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="cyan", lw=1.3))
            axes[c].set_xticks([]); axes[c].set_yticks([])
            axes[c].set_title(f"cam {ch}", fontsize=8)
        who = "config-D says CONTACT, X3D says no" if r["configD"] else \
              f"X3D says CONTACT (conf {r['x3d_conf']:.2f}), config-D says no"
        fig.suptitle(f"{scene}  frame {r['fi']}  —  {who}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"f{r['fi']:05d}.png", dpi=85)
        plt.close(fig)
    print(f"  {scene}: {sum(1 for r in recs if not r['agree'])} strips -> {out}")

out_path = _ROOT / "reports" / "auto_label_proposals.json"
slim = {sc: [{k: r[k] for k in ("fi", "old_label", "configD", "x3d", "x3d_conf",
                                "x3d_warmup", "agree", "proposed")} for r in recs]
        for sc, recs in result.items()}
out_path.write_text(json.dumps(slim, indent=1))
print(f"\nwrote {out_path}")
