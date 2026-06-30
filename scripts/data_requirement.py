"""Data-requirement estimate — how many positives to measure F1 reliably?

The CV showed config-D's F1 has a huge spread (fold std ±24pt). This script
quantifies the confidence interval at the current data size by bootstrapping the
pooled config-D CV predictions, then extrapolates how many test positives (and
distinct scenes) we'd need to estimate F1 tightly enough that a 65% claim is
distinguishable from 60% / 55%.

Two bootstraps:
  * frame-level  : resample the 1108 pool frames with replacement (binomial
                   sampling noise floor).
  * scene-level  : resample whole scenes with replacement (captures the
                   between-scene variance that actually dominates — only 4
                   contact scenes).

Outputs reports/data_requirement_results.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import eval_waveshare_contact as geo
import eval_waveshare_contact_v3_variants as vv
import eval_waveshare_contact_cv_motion as cvm
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.metrics import binary_confusion_matrix

RAW_CKPT = cvm.RAW_CKPT
OUT_JSON = _ROOT / "reports" / "data_requirement_results.json"
B = 3000
RNG = np.random.default_rng(0)


def _f1(yt, yp):
    cm = binary_confusion_matrix(list(yt), list(yp))
    return cm.f1


def main():
    print("=" * 78)
    print("  DATA-REQUIREMENT ESTIMATE (bootstrap of config-D pooled CV)")
    print("=" * 78)
    idx = DatasetIndex(geo.DATASET_ROOT, sensor_profile=cvm.PROFILE)
    ssd = MobileNetSSDDetector.load(RAW_CKPT)
    pre = geo.fit_preprocessors(idx)
    labels = geo._load_contact_labels(geo.CONTACT_CSV)

    print("  building pool + base signal ...", flush=True)
    scenes = cvm.build_pool(idx, ssd, pre, labels)
    flat = [r for recs in scenes.values() for r in recs]
    dummy_x3d = {id(r): 0.0 for r in flat}      # rule ignores X3D; skip the X3D pass
    cvm.precompute(scenes, dummy_x3d)
    for recs in scenes.values():
        cvm.fold_assign(recs)

    # Collect pooled config-D rule predictions (each frame tested once)
    pooled = []   # (scene, label, pred)
    for q in range(cvm.N_FOLDS):
        pred = cvm.eval_rule(scenes, q, (q - 1) % cvm.N_FOLDS)
        for recs in scenes.values():
            for r in recs:
                if r["q"] == q:
                    pooled.append((r["scene"], r["label"], pred[id(r)]))
    yt = np.array([p[1] for p in pooled]); yp = np.array([p[2] for p in pooled])
    n_pos = int(yt.sum())
    base_f1 = _f1(yt, yp)
    print(f"  pooled: {len(pooled)} frames, {n_pos} positive,  config-D F1 = {base_f1:.1%}")

    # ---- frame-level bootstrap ----
    n = len(pooled)
    fb = []
    for _ in range(B):
        idxs = RNG.integers(0, n, n)
        fb.append(_f1(yt[idxs], yp[idxs]))
    fb = np.array(fb)
    f_lo, f_hi = np.percentile(fb, [2.5, 97.5])
    f_hw = (f_hi - f_lo) / 2

    # ---- scene-level bootstrap ----
    by_scene = defaultdict(list)
    for s, l, p in pooled:
        by_scene[s].append((l, p))
    scene_names = list(by_scene)
    sb = []
    for _ in range(B):
        chosen = RNG.choice(len(scene_names), len(scene_names))
        yl, yq = [], []
        for ci in chosen:
            for l, p in by_scene[scene_names[ci]]:
                yl.append(l); yq.append(p)
        sb.append(_f1(yl, yq))
    sb = np.array(sb)
    s_lo, s_hi = np.percentile(sb, [2.5, 97.5])
    s_hw = (s_hi - s_lo) / 2

    print("\n  95% CONFIDENCE INTERVALS (current data):")
    print(f"    frame-level : {base_f1:.1%}  CI [{f_lo:.1%}, {f_hi:.1%}]  (±{f_hw:.1%})")
    print(f"    scene-level : {base_f1:.1%}  CI [{s_lo:.1%}, {s_hi:.1%}]  (±{s_hw:.1%})  <- dominant")

    # ---- extrapolation (frame noise floor): half-width ~ k / sqrt(n_pos) ----
    k = f_hw * np.sqrt(n_pos)
    print("\n  Frame-noise extrapolation  (CI half-width ≈ k / √positives):")
    print(f"    current: {n_pos} positives → ±{f_hw:.1%}")
    for target in (0.05, 0.03, 0.025):
        need = (k / target) ** 2
        print(f"    to reach ±{target:.1%}: ~{need:.0f} test positives "
              f"({need/n_pos:.1f}× current)")

    # scene extrapolation (rough): scene-level half-width ~ c / sqrt(n_scenes_contact)
    n_contact_scenes = sum(1 for s in by_scene if any(l for l, _ in by_scene[s]))
    print(f"\n  Scene-level variance is the binding constraint:")
    print(f"    contact-bearing scenes now: {n_contact_scenes}")
    print(f"    scene-level CI (±{s_hw:.1%}) is ~{s_hw/max(f_hw,1e-9):.1f}× the frame-level CI —")
    print(f"    i.e. diversity (more distinct contact scenarios), not just more frames,")
    print(f"    is what must grow. Rough target: 3-4× the contact scenes (~{n_contact_scenes*3}-{n_contact_scenes*4}).")

    print("\n  Verdict:")
    print(f"    To distinguish 65% from 60% (need ±<2.5%), the frame floor alone wants")
    print(f"    ~{(k/0.025)**2:.0f} positives (~{(k/0.025)**2/n_pos:.0f}× today); the scene-level CI is")
    print(f"    far wider, so the real need is MANY more distinct contact recordings.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_frames": len(pooled), "n_pos": n_pos, "config_d_f1": base_f1,
        "frame_ci": [float(f_lo), float(f_hi)], "frame_halfwidth": float(f_hw),
        "scene_ci": [float(s_lo), float(s_hi)], "scene_halfwidth": float(s_hw),
        "k_frame": float(k), "n_contact_scenes": int(n_contact_scenes),
        "positives_for_pm2_5pct": float((k / 0.025) ** 2),
        "positives_for_pm5pct": float((k / 0.05) ** 2),
    }, indent=2, default=float))
    print(f"\n  JSON → {OUT_JSON}")


if __name__ == "__main__":
    main()
