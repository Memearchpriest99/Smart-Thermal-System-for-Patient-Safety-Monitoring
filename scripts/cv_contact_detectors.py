#!/usr/bin/env python3
"""Leave-one-scene-out cross-validation for the contact detectors.

WHY THIS REPLACES THE SINGLE-SPLIT NUMBERS
------------------------------------------
The corpus contains 246 contact-positive frames spread over exactly FIVE
scenes (2ppl_fight 50, 2ppl_hug 19, 3pplhedroncolider 53, 3pp_surprise 66,
2men_clash 58). The fixed CONTACT_TEST_SCENES split locks three of those five
into test permanently, leaving two positive scenes to cover both training and
validation -- which is why ThermoX3D's recall collapses from 100% on
validation to 36% on test, and why every single-split contact number in this
project has been fragile.

At n=5 scenes the statistically defensible answer is leave-one-scene-out CV:
each positive scene is the test fold exactly once, every other scene (positive
and negative alike) trains/tunes. That uses all 246 positives instead of ~124,
and -- more importantly -- it reports a SPREAD. The project already has direct
evidence that the spread is the finding: config-D's honest 4-fold CV was
44.3% pooled with per-fold F1 of 32/51/65/0%, against a flattering 60.8% on
one split.

DETECTORS COMPARED
------------------
  * RBTCT       -- rule-based, training-free (thermal_algorithms/contact_detection/rbtct.py)
  * ThermoX3D   -- the trained 3-D CNN
  * OR-ensemble -- alarm if EITHER fires, the posture recommended by the
                   earlier investigation but never actually measured

Reported per fold and as a pooled aggregate (all folds' frames concatenated),
which is the honest headline: a mean-of-per-fold-F1 over folds with wildly
different positive counts would over-weight the small scenes.

Outputs reports/contact_cv_results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.contact_detection.rbtct import RBTCTDetector
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.augment import AugmentParams, augment_examples
from thermal_algorithms.training.balance import balance_windows
from thermal_algorithms.training.datasets import ContactFrameDataset
from thermal_algorithms.training.metrics import binary_confusion_matrix

DATA_ROOT = _REPO_ROOT.parent / "data" / "waveshare_work"
RAW_SSD = _REPO_ROOT / "checkpoints" / "mobilenet_ssd_detector" / "Waveshare_26984_rawD.thalg"
OUT_JSON = _REPO_ROOT / "reports" / "contact_cv_results.json"
CHANNELS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Cache: decode every scene once (raw frames, labels, SSD boxes, residuals)
# ---------------------------------------------------------------------------

def build_cache(idx, ssd, verbose=True):
    """{scene: {'raw': [ (f0,f1,f2) ], 'y': [...], 'boxes': [...], 'resid': [...] }}"""
    ds_all = ContactFrameDataset(idx)
    scenes = sorted({s.scene for s in ds_all.sessions})

    # Tateno background per channel, fit on the empty room (residual path).
    empty = idx.find("empty_room")
    pre = {}
    for ch in CHANNELS:
        calib = [_get(empty, ch, i) for i in range(0, empty.n_frames, 2)]
        pre[ch] = TatenoPipeline(WAVESHARE_26984).fit(calib)

    cache = {}
    for scene in scenes:
        ds = ContactFrameDataset(idx, scenes={scene})
        if len(ds) == 0:
            continue
        t0 = time.time()
        raw, y, boxes, resid = [], [], [], []
        for triplet, ev in ds:
            raw.append(triplet)
            y.append(1 if ev.any_contact else 0)
            boxes.append(tuple(ssd.predict(triplet[c]) for c in CHANNELS))
            resid.append(tuple(pre[c].predict(triplet[c]) for c in CHANNELS))
        cache[scene] = {"raw": raw, "y": y, "boxes": boxes, "resid": resid}
        if verbose:
            print(f"    {scene:<22} {len(y):>5} frames, {sum(y):>4} pos  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return cache


def _get(session, ch, fi):
    from thermal_algorithms.core.types import Frame
    arr = session.load_frames(ch)
    return Frame(data=arr[fi].astype(np.float32),
                 timestamp=fi / WAVESHARE_26984.sample_rate_hz, camera_id=ch)


# ---------------------------------------------------------------------------
# RBTCT fold
# ---------------------------------------------------------------------------

def rbtct_fold(cache, train_scenes, test_scene):
    det = RBTCTDetector(WAVESHARE_26984)

    def raw_stream(scene):
        det.reset()
        return [det.raw_decision(cache[scene]["resid"][i], cache[scene]["boxes"][i])
                for i in range(len(cache[scene]["y"]))]

    streams = [(raw_stream(s), cache[s]["y"]) for s in train_scenes]
    det.calibrate_temporal(streams)

    raw = raw_stream(test_scene)
    det.reset()
    pred = [det._gate(v) for v in raw]
    return pred, {"attack_frames": det._attack, "release_frames": det._release}


# ---------------------------------------------------------------------------
# ThermoX3D fold
# ---------------------------------------------------------------------------

def x3d_fold(cache, train_scenes, test_scene, *, gnorm, epochs, aug_copies, seed):
    det = ThermoX3DDetector(sensor_profile=WAVESHARE_26984, T=5, n_epochs=1,
                            batch_size=16, learning_rate=1e-4, weight_decay=4.6e-6,
                            min_positive_frames=2, weight_init="identity",
                            random_state=seed)

    def gn(scene):
        return [(tuple(gnorm.predict(f) for f in trip), ev)
                for trip, ev in zip(cache[scene]["raw"], _events(cache[scene]["y"]))]

    # normalisation from TRAIN scenes only
    vals = []
    for s in train_scenes:
        for trip, _e in gn(s)[:80]:
            for f in trip:
                vals.append(f.data.ravel())
    arr = np.concatenate(vals).astype(np.float32)
    det.set_normalization(float(arr.mean()), float(arr.std()) + 1e-6)

    aug = AugmentParams(flip_prob=0.48, rotate_prob=0.83,
                        max_rotation_deg=15.0, noise_sigma=1.08)
    rng = np.random.default_rng(seed)
    windows = []
    for s in train_scenes:
        ex_raw = list(zip(cache[s]["raw"], _events(cache[s]["y"])))
        windows += det._build_training_windows(gn(s))
        for _ in range(aug_copies):
            a = augment_examples(ex_raw, aug, rng=rng)
            windows += det._build_training_windows(
                [(tuple(gnorm.predict(f) for f in trip), ev) for trip, ev in a])
    windows = balance_windows(windows, seed=seed)
    n_pos = sum(1 for w in windows if w[1] == 1)
    if not windows or n_pos == 0 or n_pos * 2 != len(windows):
        return None, {"skipped": "could not balance training windows"}

    for _ in range(epochs):
        det._fit_windows(windows)

    # score the held-out scene, streaming through predict()
    det.reset()
    det._conf_threshold, det._persistence_frames = 0.5, 1
    pred = []
    for trip in cache[test_scene]["raw"]:
        ev = det.predict(tuple(gnorm.predict(f) for f in trip))
        pred.append(1 if ev.any_contact else 0)
    return pred, {"n_train_windows": len(windows), "epochs": epochs}


def _events(ys):
    from thermal_algorithms.core.types import ContactEvent
    return [ContactEvent(actors=(), pairs_in_contact=((0, 1),) if v else (),
                         timestamp=float(i), confidence=1.0) for i, v in enumerate(ys)]


# ---------------------------------------------------------------------------

def _cm_dict(cm):
    return {"accuracy": cm.accuracy, "precision": cm.precision, "recall": cm.recall,
            "f1": cm.f1, "false_alarm_rate": cm.false_alarm_rate,
            "tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn, "n": cm.total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--aug-copies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-x3d", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    idx = DatasetIndex(Path(args.data_root), sensor_profile=WAVESHARE_26984, fps=8.0)
    if not RAW_SSD.is_file():
        raise SystemExit(f"raw SSD checkpoint missing: {RAW_SSD}\n"
                         "run scripts/eval_config_d.py once to train it.")
    ssd = MobileNetSSDDetector.load(RAW_SSD)
    gnorm = GlobalNormPreprocessor(WAVESHARE_26984); gnorm.fit([])

    print("=" * 78)
    print("  Leave-one-scene-out CV -- contact detectors")
    print("=" * 78)
    print("  building cache (SSD boxes + Tateno residuals, once per scene) ...", flush=True)
    cache = build_cache(idx, ssd)

    pos_scenes = [s for s in cache if sum(cache[s]["y"]) > 0]
    all_scenes = sorted(cache)
    print(f"\n  positive scenes (folds): {sorted(pos_scenes)}")
    print(f"  total frames={sum(len(cache[s]['y']) for s in all_scenes)} "
          f"positives={sum(sum(cache[s]['y']) for s in all_scenes)}")

    folds, pooled = [], {"RBTCT": ([], []), "ThermoX3D": ([], []), "OR-ensemble": ([], [])}
    for test_scene in sorted(pos_scenes):
        train = [s for s in all_scenes if s != test_scene]
        y = cache[test_scene]["y"]
        print(f"\n  --- fold: test={test_scene} ({len(y)} frames, {sum(y)} pos) ---", flush=True)

        t0 = time.time()
        r_pred, r_info = rbtct_fold(cache, [s for s in train if sum(cache[s]['y']) > 0], test_scene)
        r_cm = binary_confusion_matrix(y, r_pred)
        print(f"    RBTCT       F1={r_cm.f1:.3f} P={r_cm.precision:.3f} R={r_cm.recall:.3f} "
              f"({r_info}) [{time.time()-t0:.0f}s]", flush=True)

        entry = {"test_scene": test_scene, "n": len(y), "n_pos": sum(y),
                 "RBTCT": _cm_dict(r_cm), "rbtct_params": r_info}
        pooled["RBTCT"][0].extend(y); pooled["RBTCT"][1].extend(r_pred)

        if not args.skip_x3d:
            t0 = time.time()
            x_pred, x_info = x3d_fold(cache, train, test_scene, gnorm=gnorm,
                                      epochs=args.epochs, aug_copies=args.aug_copies,
                                      seed=args.seed)
            if x_pred is None:
                print(f"    ThermoX3D   SKIPPED ({x_info})", flush=True)
            else:
                x_cm = binary_confusion_matrix(y, x_pred)
                print(f"    ThermoX3D   F1={x_cm.f1:.3f} P={x_cm.precision:.3f} "
                      f"R={x_cm.recall:.3f} [{time.time()-t0:.0f}s]", flush=True)
                entry["ThermoX3D"] = _cm_dict(x_cm)
                pooled["ThermoX3D"][0].extend(y); pooled["ThermoX3D"][1].extend(x_pred)

                o_pred = [1 if (a or b) else 0 for a, b in zip(r_pred, x_pred)]
                o_cm = binary_confusion_matrix(y, o_pred)
                print(f"    OR-ensemble F1={o_cm.f1:.3f} P={o_cm.precision:.3f} "
                      f"R={o_cm.recall:.3f}", flush=True)
                entry["OR-ensemble"] = _cm_dict(o_cm)
                pooled["OR-ensemble"][0].extend(y); pooled["OR-ensemble"][1].extend(o_pred)
        folds.append(entry)

    print("\n" + "=" * 78)
    print("  POOLED (all folds' frames concatenated)")
    print("=" * 78)
    summary = {}
    for name, (yt, yp) in pooled.items():
        if not yt:
            continue
        cm = binary_confusion_matrix(yt, yp)
        per = [f[name]["f1"] for f in folds if name in f]
        summary[name] = {"pooled": _cm_dict(cm), "per_fold_f1": per,
                         "mean_f1": float(np.mean(per)) if per else None,
                         "std_f1": float(np.std(per)) if per else None}
        print(f"  {name:<12} pooled F1={cm.f1:.3f} P={cm.precision:.3f} R={cm.recall:.3f} "
              f"FAR={cm.false_alarm_rate:.3f} | per-fold F1 "
              f"{np.mean(per):.3f} +/- {np.std(per):.3f}  {[round(v,2) for v in per]}")

    payload = {"protocol": "leave-one-scene-out over all contact-positive scenes",
               "folds": folds, "summary": summary,
               "epochs": args.epochs, "aug_copies": args.aug_copies, "seed": args.seed}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
    print(f"\n  saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
