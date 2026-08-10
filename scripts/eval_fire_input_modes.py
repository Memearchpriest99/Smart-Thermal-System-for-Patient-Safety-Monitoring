#!/usr/bin/env python3
"""Fire detectors x input representation: RAW vs TatenoPipeline vs GlobalNormPreprocessor.

Question: every published fire number pairs OtsuFireDetector with RAW frames and
FireSVMDetector with GlobalNorm residuals (an asymmetry disclosed in the report but
never measured). This script crosses both detectors with all three input
representations on the SAME held-out test set every prior fire result used.

TEST SET -- PINNED, and this matters
-----------------------------------
`build_task_split(..., task="fire")` is a *random* session-level draw. It was run
historically over 17 labeled scenes and produced:

    1person cig, 1ppl_single_forever, man_light_cig        (14 train / 3 test)

`waveshare_work` now holds 18 labeled scenes (`2men_clash` was integrated later), and
the same call over 18 sessions returns a COMPLETELY DIFFERENT draw
(1_man_run, 3pp_surprise, 3pplhedroncolider, man_light_cig_1). So this script pins the
historical 3 scenes explicitly and asserts they are reproducible from the 17-session
list, rather than calling build_task_split on today's index and silently scoring a
different test set than every number it is being compared against. `2men_clash` is
excluded entirely (neither train nor test) for the same reason.

Test set: 1893 frame-slots, 559 fire-positive (29.5%).
Train set: 14 scenes, 6366 frame-slots.

ROWS
----
Per input mode (raw / tateno / global_norm):
  1. OtsuFireDetector, shipped thresholds (t_ign=41.5 C)   -- "as-configured"
  2. OtsuFireDetector, t_ign recalibrated for THAT mode    -- fair-to-Otsu
  3. FireSVMDetector refit on that mode's train frames     -- fair-to-SVM
  4. FireSVMDetector, published checkpoint (fit on GlobalNorm), scored in that mode
     -- quantifies the train/serve mismatch risk flagged in demo/engine.py:108
Plus mode-independent always-alarm / never-alarm baselines, because the 29.5% base
rate makes F1 alone misleading (see reports/contact_cv_results.json for prior art on
that trap).

CAVEATS
-------
* Row 3's FireSVM is trained on waveshare train scenes ONLY, not the full corpus
  (waveshare + synth subsample) the published checkpoint used. Rows 1-3 are
  internally comparable across modes; row 3 is NOT directly comparable to the
  published full-corpus number. Row 4 IS the published checkpoint.
* TatenoPipeline is calibrated PER CAMERA here (each camera sees a different part of
  the room, so one mixed-channel background is a real handicap). The published Task-2
  GlobalNorm-vs-Tateno comparison used a single mixed-channel fit, so these Tateno
  rows are, if anything, more favourable to Tateno than that comparison was.
* Only t_ign is recalibrated in row 2. t_fire/a_limit were found empirically inert at
  the validated t_ign on raw frames (scripts/calibrate_otsu_thresholds.py); that
  finding is assumed to carry over and is NOT re-verified per mode.

Outputs reports/fire_input_modes_eval.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import reduce
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.checkpoints import CheckpointRegistry
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import FireLevel, Frame
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
from thermal_algorithms.fire_detection.otsu_pipeline import (
    OtsuFireDetector,
    _despike,
)
from thermal_algorithms.preprocessing import GlobalNormPreprocessor, TatenoPipeline
from thermal_algorithms.training import DatasetIndex
from thermal_algorithms.training.datasets import FireFrameDataset
from thermal_algorithms.training.metrics import BinaryConfusionMatrix
from thermal_algorithms.training.split import build_task_split
from thermal_algorithms.training.trainer import Trainer

DATA_ROOT = _REPO_ROOT.parent / "data" / "waveshare_work"
FULL_CORPUS_CKPT = _REPO_ROOT / "checkpoints_full_corpus"
OUT_JSON = _REPO_ROOT / "reports" / "fire_input_modes_eval.json"

# Pinned historical draw -- see module docstring.
FIRE_TEST_SCENES = frozenset({"1person cig", "1ppl_single_forever", "man_light_cig"})
LATE_ADDITION = "2men_clash"          # excluded from both splits
CALIB_SCENES = ("calibrate_room", "empty_room")   # for TatenoPipeline's background
MODES = ("raw", "tateno", "global_norm")


# ---------------------------------------------------------------------------
# Preprocessor plumbing
# ---------------------------------------------------------------------------

class PerCameraTateno:
    """TatenoPipeline calibrated independently per camera_id.

    Trainer only ever calls `.predict(frame)`, so this duck-types a Preprocessor
    without inheriting the ABC (it holds three fitted pipelines, not one state).
    """

    name = "tateno_per_camera"

    def __init__(self, profile, calib_by_cam: dict[int, list[Frame]]) -> None:
        self._pipes: dict[int, TatenoPipeline] = {}
        for cam, frames in sorted(calib_by_cam.items()):
            if not frames:
                raise ValueError(f"No calibration frames for camera {cam}.")
            self._pipes[cam] = TatenoPipeline(sensor_profile=profile).fit(frames)
        self.n_calib = {c: len(f) for c, f in sorted(calib_by_cam.items())}

    def predict(self, X: Frame) -> Frame:
        pipe = self._pipes.get(X.camera_id)
        if pipe is None:
            raise KeyError(f"No Tateno background calibrated for camera {X.camera_id}.")
        return pipe.predict(X)


def build_preprocessors(index: DatasetIndex) -> dict:
    """One preprocessor per mode; `raw` is None (Trainer skips preprocessing)."""
    calib_by_cam: dict[int, list[Frame]] = {0: [], 1: [], 2: []}
    for frame, _alert in FireFrameDataset(index, scenes=set(CALIB_SCENES)):
        calib_by_cam.setdefault(frame.camera_id, []).append(frame)

    tateno = PerCameraTateno(WAVESHARE_26984, calib_by_cam)
    print(f"  TatenoPipeline calibrated per camera on {CALIB_SCENES}: "
          f"{tateno.n_calib} frames")

    gn = GlobalNormPreprocessor(sensor_profile=WAVESHARE_26984).fit()
    return {"raw": None, "tateno": tateno, "global_norm": gn}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pool(results) -> dict:
    """Pool per-session ScenarioResults into one metric dict.

    IoU is averaged weighted by each session's ground-truth-positive count
    (tp+fn), matching evaluate_fire_detection's own "IoU over gt-positive frames
    only" convention -- an unweighted mean of per-session means would
    over-weight a 6-positive-frame scene against a 400-positive-frame one.
    """
    cm = reduce(lambda a, b: a + b, (r.confusion for r in results))
    gt_pos_per = [(r.confusion.tp + r.confusion.fn) for r in results]
    den = sum(gt_pos_per)
    iou = (sum(r.mean_iou * n for r, n in zip(results, gt_pos_per)) / den) if den else 0.0

    tp, tn, fp, fn = cm.tp, cm.tn, cm.fp, cm.fn
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    mcc_den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / mcc_den) if mcc_den > 0 else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": cm.accuracy,
        "precision": cm.precision,
        "recall": cm.recall,
        "f1": cm.f1,
        "specificity": spec,
        "balanced_accuracy": (cm.recall + spec) / 2.0,
        "mcc": mcc,
        "false_alarm_rate": cm.fp / (cm.fp + cm.tn) if (cm.fp + cm.tn) else 0.0,
        "mean_iou": iou,
        "per_scene": {r.name: {"tp": r.confusion.tp, "tn": r.confusion.tn,
                               "fp": r.confusion.fp, "fn": r.confusion.fn,
                               "f1": r.f1, "mean_iou": r.mean_iou}
                      for r in results},
    }


def constant_baseline(test_ds, always: bool) -> dict:
    """always-alarm / never-alarm, pooled the same way (IoU is 0 by construction)."""
    per_scene, cms = {}, []
    for session, examples in test_ds.by_session():
        y = [1 if a.level != FireLevel.SAFE else 0 for _f, a in examples]
        p = [1 if always else 0] * len(y)
        cm = BinaryConfusionMatrix(
            tp=sum(1 for t, q in zip(y, p) if t == 1 and q == 1),
            tn=sum(1 for t, q in zip(y, p) if t == 0 and q == 0),
            fp=sum(1 for t, q in zip(y, p) if t == 0 and q == 1),
            fn=sum(1 for t, q in zip(y, p) if t == 1 and q == 0),
        )
        cms.append(cm)
        per_scene[session.scene] = {"tp": cm.tp, "tn": cm.tn, "fp": cm.fp, "fn": cm.fn,
                                    "f1": cm.f1, "mean_iou": 0.0}

    class _R:  # minimal ScenarioResult stand-in for pool()
        def __init__(self, name, cm):
            self.name, self.confusion, self.mean_iou = name, cm, 0.0
            self.f1 = cm.f1

    out = pool([_R(n, c) for n, c in zip(per_scene, cms)])
    return out


# ---------------------------------------------------------------------------
# Otsu t_ign recalibration (per mode, on TRAIN scenes only)
# ---------------------------------------------------------------------------

def materialize(index: DatasetIndex, scenes: set[str], prep, stride: int = 1):
    """[(scene, preprocessed Frame, FireAlert)] so candidate sweeps don't
    re-run preprocessing once per candidate."""
    out = []
    for session, examples in FireFrameDataset(index, scenes=scenes).by_session():
        for i, (frame, alert) in enumerate(examples):
            if i % stride:
                continue
            f = prep.predict(frame) if prep is not None else frame
            out.append((session.scene, f, alert))
    return out


def sweep_t_ign(train_rows, candidates, min_precision: float) -> tuple[float, list[dict]]:
    """Score each candidate t_ign with the REAL stateful detector (reset per scene).

    Objective mirrors scripts/calibrate_otsu_thresholds.py: maximize F1 subject to
    precision >= min_precision. An unconstrained F1 search degenerates to alarming on
    nearly every frame (F1 never sees true negatives). Falls back to best balanced
    accuracy if no candidate clears the precision floor -- which is the expected
    outcome for a residual mode where an absolute-temperature rule cannot work.
    """
    grid = []
    for t in candidates:
        det = OtsuFireDetector(WAVESHARE_26984, t_ign=float(t))
        y_true, y_pred, last = [], [], None
        for scene, frame, alert in train_rows:
            if scene != last:
                det.reset()
                last = scene
            y_true.append(1 if alert.level != FireLevel.SAFE else 0)
            y_pred.append(1 if det.predict(frame).level != FireLevel.SAFE else 0)
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
        tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        cm = BinaryConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        grid.append({"t_ign": float(t), "f1": cm.f1, "precision": cm.precision,
                     "recall": cm.recall, "balanced_accuracy": (cm.recall + spec) / 2.0,
                     "tp": tp, "tn": tn, "fp": fp, "fn": fn})

    ok = [g for g in grid if g["precision"] >= min_precision]
    if ok:
        best = max(ok, key=lambda g: (g["f1"], g["recall"]))
        best["selected_by"] = f"max F1 s.t. precision>={min_precision}"
    else:
        best = max(grid, key=lambda g: (g["balanced_accuracy"], g["recall"]))
        best["selected_by"] = (f"NO candidate reached precision>={min_precision}; "
                               f"fell back to max balanced accuracy")
    return best, grid


def t_ign_candidates(train_rows, n: int) -> np.ndarray:
    """Data-driven candidates from the per-frame max of the despiked data.

    Must be per-mode: raw frames live in absolute Celsius (~20-60), a GlobalNorm
    residual in |deviation from frame mean| (~0-25). A fixed Celsius grid is
    meaningless on a residual, which is the whole point of the experiment.
    """
    maxes = np.array([float(_despike(f.data.astype(np.float32)).max())
                      for _s, f, _a in train_rows])
    qs = np.linspace(0.50, 0.995, n)
    return np.unique(np.quantile(maxes, qs).round(2))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--n-tign-candidates", type=int, default=20)
    ap.add_argument("--calib-stride", type=int, default=2,
                    help="Stride over train frames during the t_ign sweep (cost is "
                         "n_candidates x n_frames).")
    ap.add_argument("--min-precision", type=float, default=0.8)
    ap.add_argument("--skip-svm-refit", action="store_true",
                    help="Otsu rows only -- skips the ~2min-per-mode SVC fits.")
    args = ap.parse_args()

    t_start = time.time()
    index = DatasetIndex(args.data_root, sensor_profile=WAVESHARE_26984)
    sessions = list(index.sessions)

    # --- Reproduce and verify the pinned historical split --------------------
    historical = [s for s in sessions if s.scene != LATE_ADDITION]
    hist_train, hist_test = build_task_split(historical, task="fire")
    if set(hist_test) != set(FIRE_TEST_SCENES):
        raise SystemExit(
            f"Pinned test set no longer reproducible from the {len(historical)}-session "
            f"list: build_task_split gave {sorted(hist_test)}, expected "
            f"{sorted(FIRE_TEST_SCENES)}. The split logic or dataset changed -- resolve "
            f"before trusting any comparison against published fire numbers."
        )
    test_scenes = set(FIRE_TEST_SCENES)
    train_scenes = set(hist_train)
    _, live_test = build_task_split(sessions, task="fire")

    print(f"dataset: {len(sessions)} labeled scenes ({LATE_ADDITION} excluded from both splits)")
    print(f"  PINNED test  ({len(test_scenes)}): {sorted(test_scenes)}")
    print(f"  train        ({len(train_scenes)}): {sorted(train_scenes)}")
    print(f"  [!] build_task_split on today's {len(sessions)} sessions would instead give "
          f"{sorted(live_test)}")

    test_ds = FireFrameDataset(index, scenes=test_scenes)
    n_test = sum(1 for _ in test_ds)
    n_pos = sum(1 for _f, a in test_ds if a.level != FireLevel.SAFE)
    print(f"  test slots={n_test} fire-positive={n_pos} ({100 * n_pos / n_test:.1f}%)")

    print("\nbuilding preprocessors...")
    preps = build_preprocessors(index)

    rows: list[dict] = []

    # --- Baselines -----------------------------------------------------------
    for label, always in (("always-alarm", True), ("never-alarm", False)):
        m = constant_baseline(test_ds, always)
        m.update(detector=label, mode="(n/a)", row=label)
        rows.append(m)
        print(f"\n[baseline] {label}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
              f"bal_acc={m['balanced_accuracy']:.3f}")

    # --- Published FireSVM checkpoint (fit on GlobalNorm) --------------------
    published = None
    try:
        published = CheckpointRegistry(FULL_CORPUS_CKPT).load(
            FireSVMDetector, WAVESHARE_26984.name)
        print(f"\nloaded published FireSVM checkpoint <- {FULL_CORPUS_CKPT}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n[warn] could not load published FireSVM checkpoint ({exc}); "
              f"skipping the mismatch rows")

    # --- Per-mode sweep ------------------------------------------------------
    for mode in MODES:
        prep = preps[mode]
        print(f"\n{'=' * 70}\nMODE: {mode}\n{'=' * 70}")

        # 1. Otsu, shipped thresholds
        det = OtsuFireDetector(WAVESHARE_26984)
        r = pool(Trainer.evaluate_fire_detection(
            det, test_ds, preprocessor=prep, mode=mode, verbose=False))
        r.update(detector="OtsuFireDetector", mode=mode, row="otsu_shipped",
                 params={"t_ign": det.get_params()["t_ign"]})
        rows.append(r)
        print(f"  [1] Otsu shipped (t_ign=41.5): f1={r['f1']:.3f} "
              f"prec={r['precision']:.3f} rec={r['recall']:.3f} "
              f"bal_acc={r['balanced_accuracy']:.3f} iou={r['mean_iou']:.3f}")

        # 2. Otsu, t_ign recalibrated for this mode on TRAIN scenes
        t0 = time.time()
        cal_rows = materialize(index, train_scenes, prep, stride=args.calib_stride)
        cands = t_ign_candidates(cal_rows, args.n_tign_candidates)
        print(f"  [2] recalibrating t_ign on {len(cal_rows)} train frames "
              f"(stride={args.calib_stride}), {len(cands)} candidates "
              f"[{cands.min():.2f} .. {cands.max():.2f}] ...")
        best, grid = sweep_t_ign(cal_rows, cands, args.min_precision)
        print(f"      -> t_ign={best['t_ign']:.2f} ({best['selected_by']}) "
              f"[{time.time() - t0:.0f}s]")
        det2 = OtsuFireDetector(WAVESHARE_26984, t_ign=best["t_ign"])
        r2 = pool(Trainer.evaluate_fire_detection(
            det2, test_ds, preprocessor=prep, mode=mode, verbose=False))
        r2.update(detector="OtsuFireDetector", mode=mode, row="otsu_recalibrated",
                  params={"t_ign": best["t_ign"], "selected_by": best["selected_by"],
                          "train_grid": grid})
        rows.append(r2)
        print(f"      f1={r2['f1']:.3f} prec={r2['precision']:.3f} "
              f"rec={r2['recall']:.3f} bal_acc={r2['balanced_accuracy']:.3f} "
              f"iou={r2['mean_iou']:.3f}")
        del cal_rows

        # 3. FireSVM refit on this mode
        if not args.skip_svm_refit:
            t0 = time.time()
            train_rows = materialize(index, train_scenes, prep, stride=1)
            X = [f for _s, f, _a in train_rows]
            y = [a for _s, _f, a in train_rows]
            print(f"  [3] refitting FireSVM on {len(X)} {mode} frames ...")
            svm = FireSVMDetector(sensor_profile=WAVESHARE_26984, class_weight="balanced")
            svm.fit(iter(X), iter(y))
            r3 = pool(Trainer.evaluate_fire_detection(
                svm, test_ds, preprocessor=prep, mode=mode, verbose=False))
            r3.update(detector="FireSVMDetector", mode=mode, row="svm_refit",
                      params={"n_train": len(X), "fit_seconds": round(time.time() - t0, 1)})
            rows.append(r3)
            print(f"      f1={r3['f1']:.3f} prec={r3['precision']:.3f} "
                  f"rec={r3['recall']:.3f} bal_acc={r3['balanced_accuracy']:.3f} "
                  f"[{time.time() - t0:.0f}s]")
            del train_rows, X, y

        # 4. Published checkpoint scored in this mode (mismatch quantification)
        if published is not None:
            r4 = pool(Trainer.evaluate_fire_detection(
                published, test_ds, preprocessor=prep, mode=mode, verbose=False))
            r4.update(detector="FireSVMDetector", mode=mode, row="svm_published_ckpt",
                      params={"trained_on": "global_norm (full corpus)",
                              "matched": mode == "global_norm"})
            rows.append(r4)
            print(f"  [4] published ckpt in {mode}: f1={r4['f1']:.3f} "
                  f"prec={r4['precision']:.3f} rec={r4['recall']:.3f} "
                  f"bal_acc={r4['balanced_accuracy']:.3f}"
                  f"{'' if mode == 'global_norm' else '   <-- MISMATCHED INPUT'}")

    # --- Emit ---------------------------------------------------------------
    payload = {
        "generated_by": "scripts/eval_fire_input_modes.py",
        "test_set": {
            "scenes": sorted(test_scenes),
            "pinned": True,
            "reason": "historical 17-session build_task_split(task='fire') draw; today's "
                      "18-session list yields a different draw",
            "todays_draw_would_be": sorted(live_test),
            "n_slots": n_test,
            "n_fire_positive": n_pos,
            "positive_rate": round(n_pos / n_test, 4),
        },
        "train_set": {"scenes": sorted(train_scenes), "excluded": [LATE_ADDITION]},
        "modes": list(MODES),
        "tateno_calibration_scenes": list(CALIB_SCENES),
        "min_precision_floor": args.min_precision,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  [{payload['elapsed_seconds']:.0f}s total]")

    # --- Table --------------------------------------------------------------
    hdr = (f"{'detector / row':<34}{'input':<13}{'acc':>7}{'prec':>7}{'rec':>7}"
           f"{'F1':>7}{'balAcc':>8}{'MCC':>7}{'FAR':>7}{'IoU':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        label = f"{r['detector']} [{r['row'].replace('_', ' ')}]"
        print(f"{label:<34}{r['mode']:<13}{r['accuracy']:>7.3f}{r['precision']:>7.3f}"
              f"{r['recall']:>7.3f}{r['f1']:>7.3f}{r['balanced_accuracy']:>8.3f}"
              f"{r['mcc']:>7.3f}{r['false_alarm_rate']:>7.3f}{r['mean_iou']:>7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
