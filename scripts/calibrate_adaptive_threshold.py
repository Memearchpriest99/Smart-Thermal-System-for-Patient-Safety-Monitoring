#!/usr/bin/env python3
"""Data-driven calibration of AdaptiveThresholdDetector's c_offset / min_area_pixels.

Background: like OtsuFireDetector was before scripts/calibrate_otsu_thresholds.py,
AdaptiveThresholdDetector is is_trainable=False with a no-op fit() --
c_offset=0.5, min_area_pixels=4 (and the other geometric filters) are
hand-picked defaults, never checked against labeled data. This script
mirrors calibrate_otsu_thresholds.py's structure closely:

1. Build a calibration set from waveshare_work's human TRAIN scenes
   (thermal_algorithms.training.datasets.FrameLevelDataset,
   class_filter=[PERSON_CLASS_ID], include_negative_frames=True) --
   synth has no bounding boxes, ever, so human detection training/
   calibration is waveshare-only (see CLAUDE.md).
2. Frames are run through GlobalNormPreprocessor first, matching
   scripts/eval_all_detectors.py's convention for HOGSVMDetector/
   MobileNetSSDDetector (preprocessor=preprocessor is passed there) --
   UNLIKE OtsuFireDetector's calibration, which deliberately uses raw
   frames because its thresholds are absolute temperatures. c_offset is a
   LOCAL contrast threshold (image - Gaussian-blurred-local-mean), which is
   invariant to a per-frame constant offset either way, so this choice
   mainly matters for keeping this detector's held-out eval consistent
   with the other two human detectors' eval convention.
3. c_offset candidates are data-driven: for each frame, compute the peak
   local contrast (image - GaussianBlur(image, block_size)).max() -- the
   same quantity _segment() thresholds against -- and spread candidates
   over where the positive/negative distributions of that quantity
   actually separate (mirrors calibrate_otsu_thresholds.py's
   data_driven_candidates on raw max-temp).
4. Grid search (c_offset x min_area_pixels) using the REAL, unmodified
   AdaptiveThresholdDetector.predict() per grid point -- deliberately NOT
   reimplementing a simplified version of _segment/_close/contour-filtering
   here (unlike Otsu's script, which could safely reuse standalone module
   functions); this detector's solidity/aspect/max-area filters live in
   predict() itself, and re-deriving them independently risks silently
   drifting from the real behaviour. Per-frame cost is cheap (small cv2
   ops on 80x62 frames), so this is fast enough without factoring.
5. Same --min-precision floor + --score choice as Otsu's script, for the
   same reason: F1 alone has a real blind spot on imbalanced data (see
   that script's docstring) -- the same risk applies here.
6. Held-out re-eval on the SAME natural-ratio human_test_scenes
   HOGSVMDetector/MobileNetSSDDetector are evaluated against, via
   evaluate_human_timed (with the preprocessor, matching their convention).

Writes:
  reports/adaptive_threshold_calibration.json  -- full grid + chosen params
  reports/adaptive_threshold_eval.json         -- held-out result
    (deliberately NOT named "full_corpus_eval_*.json" -- this detector was
    never in eval_all_detectors.py's scope and this script is its own eval
    driver, exactly like calibrate_otsu_thresholds.py's non-full_corpus_
    eval_* naming was for the natural-ratio Otsu run... except Otsu's WAS
    later folded into the full_corpus_eval_*.json glob deliberately, since
    it's now a first-class reported detector. Do the same for this one when
    it graduates from "just calibrated" to "reported": rename to
    full_corpus_eval_adaptive_threshold.json at that point, not before.)
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

import cv2
import numpy as np

from thermal_algorithms.core.sensor_profile import WAVESHARE_26984
from thermal_algorithms.core.types import Frame
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset, PERSON_CLASS_ID, build_task_split
from thermal_algorithms.training.balance import build_balanced_human_pool
from thermal_algorithms.training.eval_report import evaluate_human_timed

DATA_ROOT = _REPO_ROOT.parent / "data"
REPORTS_DIR = _REPO_ROOT / "reports"


def build_calibration_set(waveshare_index: DatasetIndex, train_scenes: set[str], preprocessor):
    """(preprocessed_data, label) pairs -- label=1 if a person is present."""
    ds = FrameLevelDataset(
        waveshare_index, scenes=train_scenes, class_filter=[PERSON_CLASS_ID],
        include_negative_frames=True,
    )
    frames, labels = [], []
    for frame, dets in ds:
        proc = preprocessor.predict(frame)
        frames.append(proc.data.astype(np.float32))
        labels.append(1 if dets else 0)
    print(f"  calibration set: {len(frames)} frames, {sum(labels)} positive")
    return frames, labels


def build_balanced_calibration_set(waveshare_index: DatasetIndex, train_scenes: set[str],
                                    preprocessor, max_total, seed: int, exclude_scenes=None):
    pool, counts = build_balanced_human_pool(
        waveshare_index, train_scenes, exclude_scenes=exclude_scenes, max_total=max_total, seed=seed,
    )
    print(f"  balanced calibration pool: {counts}")
    frames = [preprocessor.predict(frame).data.astype(np.float32) for frame, _dets in pool]
    labels = [1 if dets else 0 for _frame, dets in pool]
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


def data_driven_candidates(pos_vals: np.ndarray, neg_vals: np.ndarray, n: int) -> list[float]:
    all_vals = np.concatenate([pos_vals, neg_vals])
    lo, hi = np.percentile(all_vals, [1.0, 99.5])
    return sorted(set(np.round(np.linspace(lo, hi, n), 3).tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--n-c-candidates", type=int, default=30)
    ap.add_argument("--min-area-candidates", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32],
                     help="min_area_pixels candidates (4 is the current unvalidated default).")
    ap.add_argument("--score", choices=["f1", "recall_weighted"], default="f1")
    ap.add_argument("--min-precision", type=float, default=0.8,
                     help="Same rationale as calibrate_otsu_thresholds.py's --min-precision "
                          "-- F1 alone can pick a near-universal-positive degenerate config "
                          "on imbalanced data since it never looks at tn.")
    ap.add_argument("--balanced", action="store_true",
                     help="Resample the calibration pool to 50/50 first "
                          "(thermal_algorithms.training.balance.build_balanced_human_pool) "
                          "instead of the natural (~90%% positive) corpus ratio. Held-out "
                          "eval still runs against the unchanged, natural-ratio "
                          "human_test_scenes -- only the calibration INPUT changes.")
    ap.add_argument("--balanced-max-total", type=int, default=None)
    ap.add_argument(
        "--extra-human-test-scenes", nargs="*", default=[],
        help="Additional waveshare_work scenes to union into the held-out human_test_scenes "
             "on top of build_task_split's normal draw -- e.g. 'empty_room', to include real "
             "negative frames. Mirrors eval_all_detectors.py's flag of the same name: the "
             "normal random split for 'human' never happens to draw an all-negative scene "
             "into test, so precision/false-alarm-rate are otherwise untestable (tn=0).",
    )
    ap.add_argument("--out-calibration", default=None)
    ap.add_argument("--out-eval", default=None)
    args = ap.parse_args()
    if args.out_calibration is None:
        name = "adaptive_threshold_calibration_balanced.json" if args.balanced else "adaptive_threshold_calibration.json"
        args.out_calibration = str(REPORTS_DIR / name)
    if args.out_eval is None:
        name = "adaptive_threshold_balanced_eval.json" if args.balanced else "adaptive_threshold_eval.json"
        args.out_eval = str(REPORTS_DIR / name)

    waveshare_index = DatasetIndex(Path(args.data_root) / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    human_train_scenes, human_test_scenes = build_task_split(waveshare_index.labeled_sessions(), task="human")
    if args.extra_human_test_scenes:
        human_test_scenes = human_test_scenes | set(args.extra_human_test_scenes)
    print(f"Human train scenes: {sorted(human_train_scenes)}")
    print(f"Human test scenes: {sorted(human_test_scenes)}")

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit([])

    print("\n=== Building calibration set ===")
    if args.balanced:
        frames, labels = build_balanced_calibration_set(
            waveshare_index, human_train_scenes, preprocessor, args.balanced_max_total, seed=0,
            exclude_scenes=set(args.extra_human_test_scenes) or None,
        )
    else:
        frames, labels = build_calibration_set(waveshare_index, human_train_scenes, preprocessor)

    # block_size/morph_kernel are resolved from the sensor profile, identical
    # for every grid point -- resolve once via a throwaway reference instance.
    ref_det = AdaptiveThresholdDetector(WAVESHARE_26984)
    block_size = ref_det.block_size
    print(f"  resolved block_size={block_size}")

    def local_contrast_peak(image: np.ndarray) -> float:
        blurred = cv2.GaussianBlur(image, (block_size, block_size), 0, borderType=cv2.BORDER_REFLECT)
        return float((image - blurred).max())

    contrasts = np.array([local_contrast_peak(f) for f in frames])
    labels_arr = np.array(labels)
    pos_contrast = contrasts[labels_arr == 1]
    neg_contrast = contrasts[labels_arr == 0]
    print(f"  positive peak local contrast: min={pos_contrast.min():.2f} median={np.median(pos_contrast):.2f} "
          f"max={pos_contrast.max():.2f}")
    print(f"  negative peak local contrast: min={neg_contrast.min():.2f} median={np.median(neg_contrast):.2f} "
          f"p99={np.percentile(neg_contrast, 99):.2f} max={neg_contrast.max():.2f}")

    c_candidates = data_driven_candidates(pos_contrast, neg_contrast, args.n_c_candidates)
    print(f"\n  {len(c_candidates)} data-driven c_offset candidates spanning "
          f"[{c_candidates[0]:.2f}, {c_candidates[-1]:.2f}]")
    print(f"  min_area_pixels candidates: {args.min_area_candidates}")

    print("\n=== Grid search (real AdaptiveThresholdDetector.predict() per grid point) ===")
    grid_results = []
    t0 = time.time()
    for c_offset in c_candidates:
        for min_area in args.min_area_candidates:
            det = AdaptiveThresholdDetector(WAVESHARE_26984, c_offset=c_offset, min_area_pixels=min_area)
            y_pred = []
            for f in frames:
                dets = det.predict(Frame(data=f, timestamp=0.0, camera_id=0))
                y_pred.append(1 if dets else 0)
            cm = confusion(labels, y_pred)
            if args.score == "f1":
                score = cm["f1"]
            else:
                p, r = cm["precision"], cm["recall"]
                score = (5 * p * r / (4 * p + r)) if (p + r) else 0.0
            grid_results.append({"c_offset": c_offset, "min_area_pixels": min_area, "score": score, **cm})
        print(f"  c_offset={c_offset:.2f} done [{time.time() - t0:.1f}s elapsed]", flush=True)

    qualifying = [r for r in grid_results if r["precision"] >= args.min_precision]
    if qualifying:
        best = max(qualifying, key=lambda r: r["score"])
        print(f"\n{len(qualifying)}/{len(grid_results)} grid points have precision >= "
              f"{args.min_precision} -- selecting best {args.score} among those.")
    else:
        best = max(grid_results, key=lambda r: r["score"])
        print(f"\nWARNING: no grid point reached precision >= {args.min_precision} -- "
              f"falling back to the unconstrained best-{args.score} config.")

    print(f"\n=== Best config (precision >= {args.min_precision}, ranked by {args.score}) ===")
    print(json.dumps(best, indent=2))

    calib_payload = {
        "balanced": args.balanced,
        "score_metric": args.score,
        "n_calibration_frames": len(frames),
        "n_calibration_positive": int(labels_arr.sum()),
        "human_train_scenes": sorted(human_train_scenes),
        "block_size": block_size,
        "pos_contrast_stats": {"min": float(pos_contrast.min()), "median": float(np.median(pos_contrast)),
                                "max": float(pos_contrast.max())},
        "neg_contrast_stats": {"min": float(neg_contrast.min()), "median": float(np.median(neg_contrast)),
                                "p99": float(np.percentile(neg_contrast, 99)), "max": float(neg_contrast.max())},
        "best": best,
        "grid": grid_results,
    }
    out_calib = Path(args.out_calibration)
    out_calib.parent.mkdir(parents=True, exist_ok=True)
    with out_calib.open("w", encoding="utf-8") as fh:
        json.dump(calib_payload, fh, indent=2)
    print(f"Saved calibration grid -> {out_calib}")

    print("\n=== Held-out evaluation on human_test_scenes (real detector, with preprocessor) ===")
    det = AdaptiveThresholdDetector(
        WAVESHARE_26984, c_offset=best["c_offset"], min_area_pixels=best["min_area_pixels"],
    )
    det.fit([])
    test_ds = FrameLevelDataset(
        waveshare_index, scenes=human_test_scenes, class_filter=[PERSON_CLASS_ID],
        include_negative_frames=True,
    )
    result = evaluate_human_timed(det, test_ds, preprocessor=preprocessor, variant="fp32_baseline",
                                   detector_name="AdaptiveThresholdDetector")
    print(json.dumps(result.to_dict(), indent=2))

    out_eval = Path(args.out_eval)
    out_eval.parent.mkdir(parents=True, exist_ok=True)
    with out_eval.open("w", encoding="utf-8") as fh:
        json.dump({
            "balanced": args.balanced,
            "human_test_scenes": sorted(human_test_scenes),
            "human_train_scenes": sorted(human_train_scenes),
            "adaptive_threshold_calibrated_params": {
                "c_offset": best["c_offset"], "min_area_pixels": best["min_area_pixels"],
            },
            "calibration_note": (
                f"c_offset/min_area_pixels grid-searched (score={args.score}) against "
                f"{len(frames)} frames ({int(labels_arr.sum())} positive) from waveshare_work "
                "human-train scenes"
                + (" (50/50 BALANCED subset, thermal_algorithms.training.balance."
                   "build_balanced_human_pool)" if args.balanced else " (natural ~90% positive ratio)")
                + ". Held-out numbers below are on the SAME natural-ratio human_test_scenes "
                "HOGSVMDetector/MobileNetSSDDetector are evaluated against. See the "
                "calibration JSON for the full grid."
            ),
            "results": [result.to_dict()],
        }, fh, indent=2)
    print(f"Saved held-out eval -> {out_eval}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
