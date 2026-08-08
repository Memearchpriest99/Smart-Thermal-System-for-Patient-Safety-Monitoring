#!/usr/bin/env python3
"""Data-driven calibration of OtsuFireDetector's t_ign / t_fire / a_limit.

Background: OtsuFireDetector.fit() is a documented no-op (see its docstring)
-- t_ign=45C, t_fire=60C, a_limit~=3% of frame are EDA guesses from early in
the project, never checked against real labeled data (see CLAUDE.md "Key
open items" #6). This script actually learns them:

1. Build a calibration set the SAME way FireSVMDetector's full-corpus
   training does (thermal_algorithms.training.full_corpus.
   stream_fire_examples_subsampled): waveshare_work's fire TRAIN scenes in
   full (real bboxes) + a per-camera-subsampled draw from every synth
   session (presence-only labels) -- so the two "trained" fire detectors in
   the report are calibrated/trained on directly comparable inputs.
2. For each candidate t_ign, segment every calibration frame ONCE (hot-pixel
   connected components, mirroring OtsuFireDetector._extract_hot_blobs
   exactly) and cache each frame's blobs as (area, max_temp) pairs -- this
   is the expensive step, so it's factored out of the inner grid loop.
3. For each candidate (t_fire, a_limit) given that t_ign, classify every
   frame from the cached blobs (cheap: ignition if any blob area < a_limit;
   potential-fire if any blob with area >= a_limit has max_temp > t_fire --
   exactly OtsuFireDetector.predict()'s Stage-2 rule, evaluated stateless/
   per-frame since Stage 3's temporal growth tracker only matters for
   ACTIVE_COMBUSTION escalation, not the ignition/potential-fire threshold
   decision this script is tuning).
4. Score every grid point by accuracy/precision/recall/F1; pick the F1-best
   (a stated, overridable choice -- a false-alarm-averse deployment might
   prefer a recall-weighted pick instead, see --score).
5. Re-evaluate the winning config on the SAME held-out fire_test_scenes
   FireSVMDetector uses (thermal_algorithms.training.split.build_task_split),
   this time with the REAL stateful OtsuFireDetector.predict() (proper
   session-level temporal continuity) via evaluate_fire_timed, so the report
   number reflects genuine end-to-end detector behaviour, not the
   simplified per-frame scoring used for the grid search itself.

Writes:
  reports/otsu_threshold_calibration.json   -- full grid + chosen params
  reports/full_corpus_eval_otsu.json        -- held-out result, report-ready
    (matches the "full_corpus_eval_*.json" glob generate_full_report.py uses)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import FireLevel
from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector, _despike, _extract_hot_blobs
from thermal_algorithms.training import DatasetIndex, FireFrameDataset, build_task_split
from thermal_algorithms.training.balance import build_balanced_fire_pool
from thermal_algorithms.training.eval_report import evaluate_fire_timed
from thermal_algorithms.training.full_corpus import stream_fire_examples_subsampled

DATA_ROOT = _REPO_ROOT.parent / "data"
REPORTS_DIR = _REPO_ROOT / "reports"


def build_calibration_set(data_root: Path, waveshare_index: DatasetIndex,
                           train_scenes: set[str], per_camera_samples: int):
    """(despiked_data, label) pairs -- label=1 if gt fire-positive. Natural
    corpus ratio (whatever fire-positive fraction the sampled stream has)."""
    frames, labels = [], []
    n = 0
    t0 = time.time()
    for frame, gt in stream_fire_examples_subsampled(
        data_root, waveshare_index, per_camera_samples=per_camera_samples, scenes=train_scenes,
    ):
        frames.append(_despike(frame.data.astype(np.float32)))
        labels.append(1 if gt.level != FireLevel.SAFE else 0)
        n += 1
        if n % 5000 == 0:
            print(f"  calibration set: {n} frames ({time.time() - t0:.1f}s elapsed)", flush=True)
    print(f"  calibration set done: {n} frames, {sum(labels)} positive ({time.time() - t0:.1f}s)")
    return frames, labels


def build_balanced_calibration_set(data_root: Path, waveshare_index: DatasetIndex,
                                    train_scenes: set[str], per_camera_samples: int,
                                    max_total: int, seed: int):
    """Same (despiked_data, label) shape as build_calibration_set, but the
    underlying pool is resampled to exactly 50/50 first (balance.
    build_balanced_fire_pool) -- see thermal_algorithms/training/balance.py
    for why this is real sample-level rebalancing, not class_weight."""
    pool, counts = build_balanced_fire_pool(
        data_root, waveshare_index, train_scenes,
        synth_per_camera_samples=per_camera_samples, max_total=max_total, seed=seed,
    )
    print(f"  balanced calibration pool: {counts}")
    frames = [_despike(frame.data.astype(np.float32)) for frame, _gt in pool]
    labels = [1 if gt.level != FireLevel.SAFE else 0 for _frame, gt in pool]
    return frames, labels


def confusion(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


def data_driven_candidates(pos_max_temps: np.ndarray, neg_max_temps: np.ndarray, n: int) -> list[float]:
    """Candidate cut-points spanning the region where pos/neg max-temp
    distributions actually separate -- percentile-spaced over the union of
    both classes' observed values, not an arbitrary hand-picked range."""
    all_vals = np.concatenate([pos_max_temps, neg_max_temps])
    lo, hi = np.percentile(all_vals, [1.0, 99.5])
    return sorted(set(np.round(np.linspace(lo, hi, n), 1).tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--fire-per-camera-samples", type=int, default=1500,
                     help="Mirrors train_full_corpus.py's FireSVMDetector default -- same "
                          "calibration-input scale for both 'trained' fire detectors.")
    ap.add_argument("--n-t-candidates", type=int, default=40,
                     help="Number of t_ign/t_fire candidate cut-points (data-driven, see "
                          "data_driven_candidates()).")
    ap.add_argument("--a-limit-fractions", nargs="*", type=float,
                     default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
                     help="a_limit candidates as a fraction of frame area (0.03 is the "
                          "current unvalidated default).")
    ap.add_argument("--score", choices=["f1", "recall_weighted"], default="f1",
                     help="Grid-search selection metric. 'recall_weighted' uses F2 "
                          "(recall weighted 2x precision) -- a false-alarm-tolerant, "
                          "miss-averse choice more appropriate for a safety system; "
                          "'f1' (default) is the balanced choice.")
    ap.add_argument("--min-precision", type=float, default=0.8,
                     help="Selection is restricted to grid points with precision >= this "
                          "floor before ranking by --score; falls back to the unconstrained "
                          "best (with a loud warning) if none qualify. Needed because "
                          "F1/F2 alone have a real blind spot here: with fire-positive "
                          "frames concentrated barely above ambient (room-level event "
                          "labels applied uniformly across all 3 camera views, so many "
                          "positive frames don't show a salient hotspot to a given "
                          "camera), the unconstrained F1-optimum degenerates to predicting "
                          "fire on nearly every frame (recall~98%%, precision~28%%, "
                          "accuracy~29%%) -- F1 doesn't penalize the resulting flood of "
                          "false alarms because it never looks at tn at all. A detector "
                          "that alarms on almost every frame is operationally useless "
                          "for a safety system regardless of its F1 score.")
    ap.add_argument(
        "--balanced", action="store_true",
        help="Resample the calibration pool to exactly 50/50 fire-positive/negative "
             "(thermal_algorithms.training.balance.build_balanced_fire_pool) instead of "
             "the natural corpus ratio. Held-out eval still runs against the unchanged, "
             "natural-ratio fire_test_scenes -- only the calibration INPUT changes. "
             "Changes the default --out-calibration/--out-eval filenames (adds "
             "'_balanced') so this never collides with or overwrites the natural-ratio run.",
    )
    ap.add_argument("--balanced-max-total", type=int, default=None,
                     help="Cap on total balanced calibration examples (split evenly "
                          "pos/neg). None = use everything available after balancing.")
    ap.add_argument("--out-calibration", default=None)
    ap.add_argument("--out-eval", default=None)
    args = ap.parse_args()
    if args.out_calibration is None:
        name = "otsu_threshold_calibration_balanced.json" if args.balanced else "otsu_threshold_calibration.json"
        args.out_calibration = str(REPORTS_DIR / name)
    if args.out_eval is None:
        # Balanced eval output deliberately does NOT match "full_corpus_eval_*.json"
        # (the glob generate_full_report.py's natural-ratio section 2 uses) -- keeps
        # the balanced-training report section's inputs structurally separate, so a
        # rerun here can never silently duplicate/overwrite the natural-ratio row.
        name = "otsu_balanced_eval.json" if args.balanced else "full_corpus_eval_otsu.json"
        args.out_eval = str(REPORTS_DIR / name)

    waveshare_index = DatasetIndex(Path(args.data_root) / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    fire_train_scenes, fire_test_scenes = build_task_split(waveshare_index.labeled_sessions(), task="fire")
    print(f"Fire train scenes: {sorted(fire_train_scenes)}")
    print(f"Fire test scenes: {sorted(fire_test_scenes)}")

    print("\n=== Building calibration set (mirrors FireSVMDetector's full-corpus input) ===")
    if args.balanced:
        frames, labels = build_balanced_calibration_set(
            Path(args.data_root), waveshare_index, fire_train_scenes, args.fire_per_camera_samples,
            args.balanced_max_total, seed=0,
        )
    else:
        frames, labels = build_calibration_set(
            Path(args.data_root), waveshare_index, fire_train_scenes, args.fire_per_camera_samples,
        )
    labels_arr = np.array(labels)
    pos_max = np.array([f.max() for f, l in zip(frames, labels) if l == 1])
    neg_max = np.array([f.max() for f, l in zip(frames, labels) if l == 0])
    print(f"  positive frame max-temp: min={pos_max.min():.1f} p25={np.percentile(pos_max,25):.1f} "
          f"median={np.median(pos_max):.1f} p75={np.percentile(pos_max,75):.1f} max={pos_max.max():.1f}")
    print(f"  negative frame max-temp: min={neg_max.min():.1f} p25={np.percentile(neg_max,25):.1f} "
          f"median={np.median(neg_max):.1f} p75={np.percentile(neg_max,75):.1f} max={neg_max.max():.1f} "
          f"p99={np.percentile(neg_max,99):.1f}")

    t_candidates = data_driven_candidates(pos_max, neg_max, args.n_t_candidates)
    print(f"\n  {len(t_candidates)} data-driven t candidates spanning "
          f"[{t_candidates[0]:.1f}, {t_candidates[-1]:.1f}] C")

    w, h = WAVESHARE_26984.resolution
    a_limit_candidates = sorted({max(1, int(round(frac * w * h))) for frac in args.a_limit_fractions})
    print(f"  a_limit candidates (px): {a_limit_candidates}  (frame area = {w*h} px)")

    print("\n=== Grid search (vectorized: pad each frame's blobs to K slots) ===")
    K = 8  # generous cap on simultaneous hot blobs per frame
    n_frames = len(frames)
    grid_results = []
    best = None
    t0 = time.time()
    for t_ign in t_candidates:
        # Segment once per t_ign -- shared across every (t_fire, a_limit) below.
        areas = np.full((n_frames, K), np.inf, dtype=np.float64)
        temps = np.full((n_frames, K), -np.inf, dtype=np.float64)
        n_truncated = 0
        for i, f in enumerate(frames):
            blobs = _extract_hot_blobs(f, t_ign)  # already sorted by area desc
            if len(blobs) > K:
                n_truncated += 1
                blobs = blobs[:K]
            for j, b in enumerate(blobs):
                areas[i, j] = b["area"]
                temps[i, j] = b["max_temp"]
        if n_truncated:
            print(f"  WARNING: {n_truncated} frame(s) had >{K} blobs at t_ign={t_ign:.1f}, "
                  f"truncated to the {K} largest (area is the ignition-relevant quantity, "
                  f"so keeping the largest blobs is conservative for that check).")

        for t_fire in [t for t in t_candidates if t >= t_ign]:
            for a_limit in a_limit_candidates:
                ignition = (areas < a_limit).any(axis=1)
                potential = ((areas >= a_limit) & (temps > t_fire)).any(axis=1)
                y_pred = (ignition | potential).astype(np.int64)
                cm = confusion(labels, y_pred.tolist())
                if args.score == "f1":
                    score = cm["f1"]
                else:
                    p, r = cm["precision"], cm["recall"]
                    score = (5 * p * r / (4 * p + r)) if (p + r) else 0.0  # F2
                row = {"t_ign": t_ign, "t_fire": t_fire, "a_limit": a_limit, "score": score, **cm}
                grid_results.append(row)
        print(f"  t_ign={t_ign:.1f} done [{time.time() - t0:.1f}s elapsed]", flush=True)

    # Selection: restrict to precision >= --min-precision first (see that
    # arg's help for why unconstrained F1/F2 is unsafe here), THEN rank by
    # --score. Falls back to the unconstrained global best, loudly, if no
    # grid point clears the floor at all.
    qualifying = [r for r in grid_results if r["precision"] >= args.min_precision]
    if qualifying:
        best = max(qualifying, key=lambda r: r["score"])
        print(f"\n{len(qualifying)}/{len(grid_results)} grid points have precision >= "
              f"{args.min_precision} -- selecting best {args.score} among those.")
    else:
        best = max(grid_results, key=lambda r: r["score"])
        print(f"\nWARNING: no grid point reached precision >= {args.min_precision} -- "
              f"falling back to the unconstrained best-{args.score} config. Inspect the "
              "full grid before trusting this pick; it may be the degenerate "
              "near-universal-positive regime (see this script's docstring/--min-precision "
              "help).")

    print(f"\n=== Best config (precision >= {args.min_precision}, ranked by {args.score}) ===")
    print(json.dumps(best, indent=2))

    calib_payload = {
        "balanced": args.balanced,
        "score_metric": args.score,
        "n_calibration_frames": len(frames),
        "n_calibration_positive": int(labels_arr.sum()),
        "fire_train_scenes": sorted(fire_train_scenes),
        "pos_max_temp_stats": {
            "min": float(pos_max.min()), "median": float(np.median(pos_max)), "max": float(pos_max.max()),
        },
        "neg_max_temp_stats": {
            "min": float(neg_max.min()), "median": float(np.median(neg_max)),
            "p99": float(np.percentile(neg_max, 99)), "max": float(neg_max.max()),
        },
        "best": best,
        "grid": grid_results,
    }
    out_calib = Path(args.out_calibration)
    out_calib.parent.mkdir(parents=True, exist_ok=True)
    with out_calib.open("w", encoding="utf-8") as fh:
        json.dump(calib_payload, fh, indent=2)
    print(f"Saved calibration grid -> {out_calib}")

    print("\n=== Held-out evaluation on fire_test_scenes (real stateful OtsuFireDetector) ===")
    det = OtsuFireDetector(
        WAVESHARE_26984, t_ign=best["t_ign"], t_fire=best["t_fire"], a_limit=best["a_limit"],
    )
    det.fit([])
    test_ds = FireFrameDataset(waveshare_index, scenes=fire_test_scenes)
    result = evaluate_fire_timed(det, test_ds, preprocessor=None, variant="fp32_baseline",
                                  detector_name="OtsuFireDetector")
    print(json.dumps(result.to_dict(), indent=2))

    out_eval = Path(args.out_eval)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    with out_eval.open("w", encoding="utf-8") as fh:
        json.dump({
            "checkpoints_dir": None,
            "balanced": args.balanced,
            "fire_test_scenes": sorted(fire_test_scenes),
            "fire_train_scenes": sorted(fire_train_scenes),
            "otsu_calibrated_thresholds": {
                "t_ign": best["t_ign"], "t_fire": best["t_fire"], "a_limit": best["a_limit"],
            },
            "otsu_calibration_note": (
                f"t_ign/t_fire/a_limit grid-searched (score={args.score}) against "
                f"{len(frames)} frames ({int(labels_arr.sum())} positive) sampled from the "
                + ("50/50 BALANCED (thermal_algorithms.training.balance."
                   "build_balanced_fire_pool) subset of " if args.balanced else "")
                + "the full corpus the same way FireSVMDetector's full-corpus training is "
                "sampled (waveshare fire-train scenes in full + "
                f"{args.fire_per_camera_samples}/camera/session from every synth room, "
                "before balancing)."
                " Held-out numbers below are on the SAME natural-ratio fire_test_scenes "
                "FireSVMDetector is evaluated against -- only the calibration input "
                "changed. See the calibration JSON for the full grid."
            ),
            "results": [result.to_dict()],
        }, fh, indent=2)
    print(f"Saved held-out eval -> {out_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
