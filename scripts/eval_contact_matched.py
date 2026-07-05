"""Matched comparison: config-D (rule) vs Thermo-X3D T5v2 (ONNX) on ONE
identical in-domain test split, with confusion matrices and PC timing.

Both algorithms are scored on the SAME frames: the T5_v2 protocol test split
(per-scene 60/15/25 timeline, test = last 25%), p25 session backgrounds,
2men_clash held out. X3D confidences and p25 residuals are reused from
scripts/_x3d_backend_cache/ (built by eval_x3d_backends.py); config-D runs
raw MobileNet-SSD for boxes + the cached p25 residual for blobs, then the
two-body merge rule + temporal morphology (open 2, close 7).

Outputs reports/contact_matched_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import eval_waveshare_contact as geo
import eval_waveshare_contact_v8 as v8
import ablate_preprocessing as ab
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex

CACHE = Path(__file__).resolve().parent / "_x3d_backend_cache"
HELD_OUT = "2men_clash"
THRESHOLD = 0.35
TRAIN_FRAC, VAL_FRAC = 0.60, 0.15
LO, LC = 2, 7            # temporal morphology (open, close)
OUT = _ROOT / "reports" / "contact_matched_results.json"


def cm(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    tp = int(np.sum((yp == 1) & (yt == 1)))
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    n = tp + tn + fp + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / n if n else 0.0
    far = fp / (fp + tn) if fp + tn else 0.0
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, prec=prec, rec=rec,
                f1=f1, far=far, n=n)


def main() -> None:
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=geo.PROFILE)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)
    ssd = MobileNetSSDDetector.load(ab.RAW_CKPT)

    scene_files = sorted(CACHE.glob("*.npz"))
    if not scene_files:
        raise SystemExit("no cache; run eval_x3d_backends.py build first")

    # groups: 'home' (original scenes) vs 'arch' (06-30 interval-labeled)
    groups = {"home": {"yt": [], "cfg": [], "x3d": []},
              "arch": {"yt": [], "cfg": [], "x3d": []}}
    cfg_frame_times = []
    n_scenes = 0

    for f in scene_files:
        scene = f.stem
        if scene == HELD_OUT:
            continue
        grp = "arch" if scene.startswith("arch0630") else "home"
        conf_file = CACHE / f"confs_onnx_fp32_{scene}.npy"
        if not conf_file.exists():
            print(f"skip {scene}: no X3D confs")
            continue
        try:
            s = idx.find(scene)
        except Exception:
            print(f"skip {scene}: not in index")
            continue

        d = np.load(f)
        resid, labs = d["resid"], d["labs"]        # (3,n,H,W), (n,)
        confs = np.load(conf_file)
        n = len(labs)
        fis = sorted(fi for fi in labels[scene] if 0 <= fi < s.n_frames)
        if len(fis) != n:
            print(f"skip {scene}: cache/label length mismatch {len(fis)} vs {n}")
            continue

        n_va = int(n * (TRAIN_FRAC + VAL_FRAC))
        test_range = range(n_va, n)
        if not test_range:
            continue
        morph_start = max(0, n_va - (LC + 3))

        # config-D raw decisions over [morph_start, n)
        raw_dec = {}
        for k in range(morph_start, n):
            cams = []
            t0 = time.perf_counter()
            for c, ch in enumerate(geo.CHANNELS):
                boxes = [dd.bbox for dd in ssd.predict(geo.get_frame(s, ch, fis[k]))]
                cams.append({"boxes": boxes, "resid": resid[c, k]})
            dec = ab.v9core_decision({"cams": cams}, "resid")
            if k >= n_va:                          # time only scored frames
                cfg_frame_times.append((time.perf_counter() - t0) * 1e3)
            raw_dec[k] = dec

        # temporal morphology over the contiguous tail
        order = list(range(morph_start, n))
        morphed = v8.morph([raw_dec[k] for k in order], LO, LC)
        cfg_pred = {k: morphed[i] for i, k in enumerate(order)}

        for k in test_range:
            groups[grp]["yt"].append(int(labs[k]))
            groups[grp]["cfg"].append(int(cfg_pred[k]))
            groups[grp]["x3d"].append(1 if confs[k] > THRESHOLD else 0)

        geo._FRAMES_CACHE.clear()
        n_scenes += 1
        print(f"  {scene} [{grp}]: {len(test_range)} test frames", flush=True)

    all_yt = groups["home"]["yt"] + groups["arch"]["yt"]
    all_cfg = groups["home"]["cfg"] + groups["arch"]["cfg"]
    all_x3d = groups["home"]["x3d"] + groups["arch"]["x3d"]

    res = {
        "threshold": THRESHOLD,
        "n_scenes": n_scenes,
        "config_D_pc_ms_per_frame": float(np.median(cfg_frame_times)),
        "in_domain_all": {"config_D": cm(all_yt, all_cfg),
                          "thermo_x3d_onnx": cm(all_yt, all_x3d)},
        "in_domain_home": {"config_D": cm(groups["home"]["yt"], groups["home"]["cfg"]),
                           "thermo_x3d_onnx": cm(groups["home"]["yt"], groups["home"]["x3d"])},
        "in_domain_arch0630": {"config_D": cm(groups["arch"]["yt"], groups["arch"]["cfg"]),
                               "thermo_x3d_onnx": cm(groups["arch"]["yt"], groups["arch"]["x3d"])},
    }
    OUT.write_text(json.dumps(res, indent=2))

    for grp in ("in_domain_home", "in_domain_arch0630", "in_domain_all"):
        g = res[grp]
        n = g["config_D"]["n"]; pos = g["config_D"]["tp"] + g["config_D"]["fn"]
        print("\n" + "=" * 68)
        print(f"{grp}: {n} frames, {pos} positive")
        print(f"{'algo':<16} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'FAR':>6} "
              f"| {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        for name in ("config_D", "thermo_x3d_onnx"):
            m = g[name]
            print(f"{name:<16} {m['acc']:>6.3f} {m['prec']:>6.3f} {m['rec']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['far']:>6.3f} | {m['tp']:>5} {m['fp']:>5} "
                  f"{m['fn']:>5} {m['tn']:>5}")
    print(f"\nconfig-D PC runtime: {res['config_D_pc_ms_per_frame']:.1f} ms/frame")
    print(f"wrote {OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
