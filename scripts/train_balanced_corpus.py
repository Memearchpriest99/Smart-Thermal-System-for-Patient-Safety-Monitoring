#!/usr/bin/env python3
"""Balanced (50/50) training driver for the 5 detectors that get an actual
fit()/chunked-fit() retrain in the balanced-training pass (the other 2 in
scope, OtsuFireDetector/AdaptiveThresholdDetector, are is_trainable=False --
they get calibrated instead, via scripts/calibrate_otsu_thresholds.py
--balanced / scripts/calibrate_adaptive_threshold.py --balanced).
MVSTGCNDetector is out of scope entirely (project owner's call -- "useless,
leave as is").

Fire:    FireSVMDetector on thermal_algorithms.training.balance.
         build_balanced_fire_pool (waveshare fire-train scenes in full +
         a bounded synth draw, then resampled 50/50).
Human:   HOGSVMDetector + MobileNetSSDDetector on the SAME
         build_balanced_human_pool output (waveshare-only -- synth has no
         bounding boxes, ever; this pool is capped by the scarce negative
         class, ~800-900 frames total, so it is MUCH smaller than the
         natural-ratio training set -- a real, disclosed trade-off, not a
         bug, see thermal_algorithms/training/balance.py's docstring).
Contact: ThermoX3DDetector via a custom chunked training loop (NOT
         det.fit() directly -- see train_contact_balanced()'s docstring for
         why) built on thermal_algorithms.training.balance's
         build_balanced_contact_pools / split_contact_pools_train_val /
         size_and_cap_negative_runs / interleave_chunks (frame-level
         resampling would corrupt this detector's T-frame sliding-window
         contiguity -- see balance.py's module docstring). Fixes the
         chunked-training bug documented in memory (thermox3d_training_bug.md
         -- both prior checkpoints were degenerate constant classifiers):
         persistent AdamW optimizer, checkpoint-level (not per-chunk) frozen
         normalisation, class-mixed super-chunks with a ratio cap, a real
         train/val split with early stopping, and post-hoc threshold/
         persistence-frames recalibration. By far the most expensive part of
         this script; time-boxed with a hard --contact-time-budget-hours
         cutoff, same pattern as scripts/train_thermox3d_subset.py, and
         checkpointed incrementally.

All checkpoints register to a NEW root (--out, default checkpoints_balanced/)
-- never the same root as checkpoints_full_corpus/, so this can never
overwrite the natural-ratio weights the existing report is built on.

Usage::

    python scripts/train_balanced_corpus.py --skip-contact   # fire+human only
    python scripts/train_balanced_corpus.py --only-contact
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import logging
import random
import signal
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

LOG_DIR = _REPO_ROOT / "logs"
logger = logging.getLogger("train_balanced_corpus")


def _setup_diagnostics() -> Path:
    """Mirrors train_full_corpus.py's _setup_diagnostics -- faulthandler for
    native/silent crashes, signal logging for external stops."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    diag_path = LOG_DIR / f"train_balanced_corpus_diag_{ts}.log"
    diag_fh = open(diag_path, "a", buffering=1, encoding="utf-8")
    faulthandler.enable(file=diag_fh, all_threads=True)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(diag_fh)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    def _on_signal(signum, _frame):
        logger.warning(f"received signal {signum} ({signal.Signals(signum).name}) -- exiting")
        diag_fh.flush()
        raise SystemExit(128 + signum)

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass

    print(f"Diagnostics (crash/signal/hang traces) -> {diag_path}", flush=True)
    return diag_path


from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector  # noqa: E402
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.training import DatasetIndex, build_task_split  # noqa: E402
from thermal_algorithms.training.balance import (  # noqa: E402
    build_balanced_contact_pools,
    build_balanced_fire_pool,
    build_balanced_human_pool,
    interleave_chunks,
    size_and_cap_negative_runs,
    split_contact_pools_train_val,
)
from thermal_algorithms.training.metrics import binary_confusion_matrix

DATA_ROOT = _REPO_ROOT.parent / "data"
DEFAULT_OUT = _REPO_ROOT / "checkpoints_balanced"


# ---------------------------------------------------------------------------
# Fire
# ---------------------------------------------------------------------------

def train_fire_balanced(data_root: Path, waveshare_index: DatasetIndex, train_scenes: set[str],
                         preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry,
                         *, synth_per_camera_samples: int, max_total, seed: int) -> FireSVMDetector:
    print("\n=== Training FireSVMDetector on a balanced (50/50) pool ===")
    pool, counts = build_balanced_fire_pool(
        data_root, waveshare_index, train_scenes,
        synth_per_camera_samples=synth_per_camera_samples, max_total=max_total, seed=seed,
    )
    print(f"  balanced fire pool: {counts}")
    X = [preprocessor.predict(frame) for frame, _alert in pool]
    y = [alert for _frame, alert in pool]

    # class_weight left at None (not "balanced") -- balance_examples() already
    # resampled to exactly 50/50 at the sample level, so sklearn's loss
    # reweighting would just compute weight=1.0 for both classes anyway.
    det = FireSVMDetector(sensor_profile=WAVESHARE_26984, class_weight=None)
    t0 = time.time()
    logger.info("FireSVMDetector.fit() starting (balanced)")
    det.fit(X, y)
    logger.info("FireSVMDetector.fit() returned")
    print(f"  fit() done in {time.time() - t0:.1f}s")
    print(f"  saved -> {registry.register(det)}")
    return det


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------

def train_human_balanced(waveshare_index: DatasetIndex, train_scenes: set[str],
                          preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry,
                          *, exclude_scenes, max_total, seed: int, mobilenet_epochs) -> dict:
    print("\n=== Training human detectors on a balanced (50/50) pool ===")
    pool, counts = build_balanced_human_pool(
        waveshare_index, train_scenes, exclude_scenes=exclude_scenes, max_total=max_total, seed=seed,
    )
    print(f"  balanced human pool: {counts}")
    examples = [(preprocessor.predict(frame), dets) for frame, dets in pool]

    detectors = {}

    # class_weight="balanced" kept (NOT None): HOGSVMDetector.fit() does not
    # train one sample per frame -- it emits one positive HOG patch per
    # ground-truth bbox plus up to n_negatives_per_frame=5 random background
    # patches PER FRAME (positive or negative), so the frame-level 50/50
    # resampling above does NOT translate into a 50/50 patch-level ratio
    # (it's closer to 1:5 positive:negative in practice). Unlike
    # FireSVMDetector (genuinely one sample per frame, where class_weight is
    # provably a no-op on an even pool), dropping the loss reweighting here
    # would silently reintroduce an uncorrected imbalance.
    hog = HOGSVMDetector(sensor_profile=WAVESHARE_26984, class_weight="balanced")
    t0 = time.time()
    hog.fit(examples)
    print(f"  HOGSVMDetector fit() done in {time.time() - t0:.1f}s -- saved -> {registry.register(hog)}")
    detectors["hog_svm"] = hog

    mnet = MobileNetSSDDetector(sensor_profile=WAVESHARE_26984)
    t0 = time.time()
    mnet.fit(examples, n_epochs=mobilenet_epochs)
    print(f"  MobileNetSSDDetector fit() done in {time.time() - t0:.1f}s -- saved -> {registry.register(mnet)}")
    detectors["mobilenet_ssd"] = mnet

    return detectors


# ---------------------------------------------------------------------------
# Contact -- balanced chunked training (ThermoX3D only)
# ---------------------------------------------------------------------------
#
# Deliberately bypasses ThermoX3DDetector.fit()'s all-in-one contract (build
# windows from raw examples, train, done) in favour of a custom loop that
# calls det._fit_windows()/_eval_windows_loss() directly. Three reasons this
# script needs that extra control, not just det.fit():
#
#   1. Normalisation must be frozen (det.set_normalization(...)) from a
#      TRAIN-only sample BEFORE any windows are built -- train or val. This
#      is the ordering fix for the bug documented in memory
#      (thermox3d_training_bug.md): the old design recomputed mean/std from
#      whatever chunk fit() happened to be given, so the saved checkpoint's
#      stats reflected only the LAST chunk trained on.
#   2. Windows must be built PER RUN, never by concatenating raw frames
#      across two different runs into one list -- fit()'s own windowing
#      would otherwise produce "seam" windows straddling the boundary
#      between two physically unrelated clips.
#   3. A held-out validation set (for early stopping and threshold/
#      persistence-frames recalibration) needs its windows materialized once
#      and reused across many loss checks -- det.fit() has no concept of a
#      validation set at all.


def _decode_item(thunk):
    """thunk() -> (source_label, frames_chunk, events_chunk); returns the
    list[(triplet, event)] examples for one balance.py pool item."""
    _source, frames_chunk, events_chunk = thunk()
    return list(zip(frames_chunk, events_chunk))


def _preprocess_examples(examples, preprocessor: GlobalNormPreprocessor):
    return [(tuple(preprocessor.predict(f) for f in triplet), ev) for triplet, ev in examples]


def _compute_norm_stats(items, preprocessor: GlobalNormPreprocessor, max_items: int):
    """Mean/std over a bounded TRAIN-only sample, post-preprocessing (the
    same space det._build_training_windows normalises in) -- computed ONCE,
    frozen via det.set_normalization(), never recomputed per chunk."""
    import numpy as np
    all_vals = []
    for _n, thunk in items[:max_items]:
        examples = _preprocess_examples(_decode_item(thunk), preprocessor)
        for triplet, _ev in examples:
            for f in triplet:
                all_vals.append(f.data.ravel())
    if not all_vals:
        return 0.0, 1.0
    arr = np.concatenate(all_vals).astype(np.float32)
    return float(arr.mean()), float(arr.std()) + 1e-6


def _windows_for_items(det: ThermoX3DDetector, items, preprocessor: GlobalNormPreprocessor):
    """Builds windows PER ITEM (never concatenating raw frames across items)
    then concatenates the resulting window lists -- no cross-run seam
    windows can occur, by construction. Requires det's normalisation to
    already be frozen."""
    windows = []
    for _n, thunk in items:
        examples = _preprocess_examples(_decode_item(thunk), preprocessor)
        windows += det._build_training_windows(examples)
    return windows


def _cap_window_ratio(windows, max_ratio: int, rng: random.Random):
    """Belt-and-suspenders on top of the run-level max_ratio cap in
    interleave_chunks: if a super-chunk's own windows still skew beyond
    max_ratio (e.g. because a positive burst produced far more/fewer windows
    than its paired negative item), randomly subsample the majority-label
    windows down to the cap."""
    pos = [w for w in windows if w[1] == 1]
    neg = [w for w in windows if w[1] == 0]
    if not pos or not neg:
        return windows
    if len(pos) > max_ratio * len(neg):
        pos = rng.sample(pos, max_ratio * len(neg))
    elif len(neg) > max_ratio * len(pos):
        neg = rng.sample(neg, max_ratio * len(pos))
    combined = pos + neg
    rng.shuffle(combined)
    return combined


def _collect_val_confidence_sequences(det: ThermoX3DDetector, val_items,
                                       preprocessor: GlobalNormPreprocessor):
    """One replay pass over the validation items (buffers reset per item,
    matching scripts/eval_waveshare_contact_dl.py's per-session replay
    pattern) -- returns a list of per-item (labels, confidences) sequences,
    excluding the buffer-filling warm-up prefix. Raw confidence doesn't
    depend on conf_threshold/persistence_frames, so this only needs to run
    ONCE; the (threshold, persistence) grid sweep below replays the
    persistence-gate logic in pure Python over these saved sequences instead
    of re-running the network per grid point."""
    sequences = []
    for _n, thunk in val_items:
        det.reset()
        examples = _preprocess_examples(_decode_item(thunk), preprocessor)
        labels, confs = [], []
        for triplet, event in examples:
            pred = det.predict(triplet)
            if pred.debug.get("status") == "buffer_filling":
                continue
            labels.append(1 if event.any_contact else 0)
            confs.append(pred.confidence)
        if labels:
            sequences.append((labels, confs))
    return sequences


def _simulate_persistence(confs, threshold: float, persistence_frames: int):
    alerted, count = [], 0
    for c in confs:
        count = count + 1 if c > threshold else 0
        alerted.append(1 if count >= persistence_frames else 0)
    return alerted


def _sweep_threshold_persistence(sequences, *, thresholds, persistences, min_precision: float = 0.8):
    """Grid-search (threshold, persistence_frames), restricted to grid points
    with precision >= min_precision, ranked by F1 among those -- same
    two-step selection as scripts/calibrate_otsu_thresholds.py's
    --min-precision floor, and for the same reason: an unconstrained F1
    search has a real blind spot on a small/skewed validation set (no true
    negatives being scored heavily enough to matter), and can degenerate to
    a near-universal-positive threshold that looks perfect on val but is a
    high-false-alarm-rate operating point on real data. Falls back to the
    unconstrained best-F1 grid point, loudly, if nothing clears the floor."""
    grid = []
    for th in thresholds:
        for pf in persistences:
            yt, yp = [], []
            for labels, confs in sequences:
                yt.extend(labels)
                yp.extend(_simulate_persistence(confs, th, pf))
            cm = binary_confusion_matrix(yt, yp)
            grid.append((th, pf, cm))

    qualifying = [g for g in grid if g[2].precision >= min_precision]
    if qualifying:
        th, pf, cm = max(qualifying, key=lambda g: g[2].f1)
        print(f"    {len(qualifying)}/{len(grid)} grid points have precision >= "
              f"{min_precision} -- selecting best F1 among those.")
    else:
        th, pf, cm = max(grid, key=lambda g: g[2].f1)
        print(f"    WARNING: no grid point reached precision >= {min_precision} on the "
              f"validation set -- falling back to the unconstrained best-F1 config "
              f"(precision={cm.precision:.3f}). Treat this checkpoint's calibration as "
              f"provisional; the validation set may be too small/unrepresentative.")
    return th, pf, cm


def train_contact_balanced(data_root: Path, waveshare_index: DatasetIndex, train_scenes: set[str],
                            preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry,
                            *, target_positive_frames: int, target_negative_frames: int,
                            time_budget_hours: float, checkpoint_every: int,
                            val_fraction: float, val_every_n_chunks: int, early_stop_patience: int,
                            max_ratio: int, norm_sample_items: int, min_precision: float,
                            class_weight, seed: int, log_path: Path) -> ThermoX3DDetector:
    print("\n=== Training ThermoX3DDetector on balanced (50/50) contact runs ===")
    det = ThermoX3DDetector(sensor_profile=WAVESHARE_26984, class_weight=class_weight, n_epochs=1)
    rng = random.Random(seed)
    log_fh = open(log_path, "a", buffering=1, encoding="utf-8")

    def log_record(**kv):
        rec = {"t": time.time(), **kv}
        log_fh.write(json.dumps(rec) + "\n")

    pos_items, raw_neg_runs = build_balanced_contact_pools(
        data_root, waveshare_index, train_scenes,
        target_positive_frames=target_positive_frames,
        target_negative_frames=target_negative_frames,
        T=det._T, seed=seed,
    )
    print(f"  collected {len(pos_items)} positive items, {len(raw_neg_runs)} raw negative runs")

    train_pos, val_pos, train_neg_runs, val_neg_runs = split_contact_pools_train_val(
        pos_items, raw_neg_runs, val_fraction=val_fraction, seed=seed,
    )
    train_neg_items = size_and_cap_negative_runs(
        train_neg_runs, train_pos, T=det._T, target_negative_frames=target_negative_frames,
    )
    val_neg_items = size_and_cap_negative_runs(
        val_neg_runs, val_pos, T=det._T,
        target_negative_frames=int(target_negative_frames * val_fraction / (1 - val_fraction)),
    )
    print(f"  train: {len(train_pos)} pos items + {len(train_neg_items)} neg items")
    print(f"  val:   {len(val_pos)} pos items + {len(val_neg_items)} neg items")

    # Freeze normalisation from a bounded TRAIN-only sample BEFORE building
    # any windows -- train or val (the ordering fix; see module note above).
    mean, std = _compute_norm_stats(train_pos + train_neg_items, preprocessor, norm_sample_items)
    det.set_normalization(mean, std)
    print(f"  frozen normalisation: mean={mean:.3f} std={std:.3f}")
    log_record(event="norm_frozen", mean=mean, std=std)

    val_windows = _windows_for_items(det, val_pos + val_neg_items, preprocessor)
    n_val_pos = sum(1 for w in val_windows if w[1] == 1)
    print(f"  validation windows: {len(val_windows)} (pos={n_val_pos}, neg={len(val_windows) - n_val_pos})")
    log_record(event="val_windows_built", n_windows=len(val_windows), n_pos=n_val_pos)

    order = interleave_chunks(train_pos, train_neg_items, max_ratio=max_ratio, seed=seed)
    # Group the interleaved stream into physical super-chunks of up to
    # max_ratio items per class -- each super-chunk's windows are built per
    # item (no seams) then concatenated into one _fit_windows() call.
    chunks: list[list] = []
    cur: list = []
    cur_counts = {"pos": 0, "neg": 0}
    for cls, item in order:
        if cur and cur_counts[cls] >= max_ratio:
            chunks.append(cur)
            cur, cur_counts = [], {"pos": 0, "neg": 0}
        cur.append((cls, item))
        cur_counts[cls] += 1
        if len(cur) >= 2 * max_ratio:
            chunks.append(cur)
            cur, cur_counts = [], {"pos": 0, "neg": 0}
    if cur:
        chunks.append(cur)
    print(f"  {len(chunks)} training super-chunks queued")

    time_budget_s = time_budget_hours * 3600.0
    t0 = time.time()
    n_chunks = n_pos_frames = n_neg_frames = 0
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    stopped_reason = "all chunks processed"

    for chunk in chunks:
        elapsed = time.time() - t0
        if elapsed > time_budget_s:
            stopped_reason = f"time budget exceeded ({elapsed:.0f}s > {time_budget_s:.0f}s)"
            break

        windows = _windows_for_items(det, [item for _cls, item in chunk], preprocessor)
        windows = _cap_window_ratio(windows, max_ratio, rng)
        n_pos_w = sum(1 for w in windows if w[1] == 1)
        train_loss = det._fit_windows(windows)
        n_chunks += 1
        n_pos_frames += n_pos_w
        n_neg_frames += (len(windows) - n_pos_w)
        print(f"  chunk {n_chunks}/{len(chunks)}: {len(windows)} windows "
              f"(pos={n_pos_w}), train_loss={train_loss:.4f}, {elapsed:.1f}s elapsed", flush=True)
        log_record(event="chunk", n=n_chunks, n_windows=len(windows), n_pos=n_pos_w,
                   train_loss=train_loss, elapsed_s=elapsed)

        if n_chunks % checkpoint_every == 0:
            path = registry.register(det)
            print(f"  [checkpoint] saved after {n_chunks} chunks -> {path}", flush=True)

        if n_chunks % val_every_n_chunks == 0:
            val_loss = det._eval_windows_loss(val_windows)
            print(f"    val_loss={val_loss:.4f} (best={best_val_loss:.4f})", flush=True)
            log_record(event="val", n=n_chunks, val_loss=val_loss)
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(det._get_model()[0].state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    stopped_reason = f"early stopped (no val improvement in {early_stop_patience} checks)"
                    break

    if best_state is not None:
        model, _device = det._get_model()
        model.load_state_dict(best_state)
        print(f"  restored best-val-loss state (val_loss={best_val_loss:.4f})")
        log_record(event="restored_best_state", best_val_loss=best_val_loss)
    else:
        print("  no val checkpoint recorded (training too short) -- keeping final weights as-is")

    # Threshold + persistence-frames recalibration on val (precision-floored
    # max F1) -- early stopping picked the best LOSS, but
    # conf_threshold/persistence_frames were never retuned for the resulting
    # model.
    sequences = _collect_val_confidence_sequences(det, val_pos + val_neg_items, preprocessor)
    thresholds = [round(0.05 * k, 2) for k in range(1, 19)]
    best_th, best_pf, best_cm = _sweep_threshold_persistence(
        sequences, thresholds=thresholds, persistences=(1, 2, 3), min_precision=min_precision,
    )
    # Update BOTH the live attributes (used by predict()) AND self._params
    # (what get_params()/save() actually persists -- ThermalAlgorithm.save()
    # pickles get_params(), which returns self._params as frozen at
    # construction time, not whatever the live attributes currently are; see
    # base.py's get_params()/set_params() docstrings. Mutating only the live
    # attribute -- the mistake this comment replaces -- silently reverts to
    # the constructor default on the next load()).
    det._conf_threshold = best_th
    det._persistence_frames = best_pf
    det.set_params(conf_threshold=best_th, persistence_frames=best_pf)
    print(f"  recalibrated: conf_threshold={best_th}, persistence_frames={best_pf} "
          f"(val F1={best_cm.f1:.3f}, prec={best_cm.precision:.3f}, rec={best_cm.recall:.3f})")
    log_record(event="recalibrated", conf_threshold=best_th, persistence_frames=best_pf,
               val_f1=best_cm.f1, val_precision=best_cm.precision, val_recall=best_cm.recall)

    path = registry.register(det)
    print(f"\nThermoX3DDetector done ({stopped_reason}) -- {n_chunks} chunks, "
          f"{n_pos_frames} positive / {n_neg_frames} negative windows -- saved -> {path}")
    log_record(event="done", stopped_reason=stopped_reason, n_chunks=n_chunks,
               n_pos_windows=n_pos_frames, n_neg_windows=n_neg_frames, checkpoint=str(path))
    log_fh.close()
    return det


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fire-synth-per-camera-samples", type=int, default=1500,
                     help="Mirrors train_full_corpus.py's FireSVMDetector default, applied "
                          "BEFORE balancing (so the pre-balance pool has the same scale).")
    ap.add_argument("--fire-max-total", type=int, default=None)
    ap.add_argument("--human-max-total", type=int, default=None,
                     help="Cap on the balanced human pool. None = use everything available "
                          "(capped by the scarce negative class regardless, ~800-900 frames "
                          "total -- see build_balanced_human_pool).")
    ap.add_argument(
        "--exclude-human-scenes", nargs="*", default=["empty_room"],
        help="Scenes to exclude from the balanced human training pool -- MUST match whatever "
             "--extra-human-test-scenes is passed to eval_all_detectors.py/"
             "calibrate_adaptive_threshold.py for this checkpoint's held-out eval, or those "
             "frames leak into both train and test. Defaults to ['empty_room'] since that's "
             "the standard extra negative test scene used elsewhere in this project.",
    )
    ap.add_argument("--mobilenet-epochs", type=int, default=None,
                     help="None = MobileNetSSDDetector's own default (50).")

    ap.add_argument("--contact-target-positive-frames", type=int, default=25000)
    ap.add_argument("--contact-target-negative-frames", type=int, default=25000)
    ap.add_argument("--contact-time-budget-hours", type=float, default=5.0)
    ap.add_argument("--contact-checkpoint-every", type=int, default=5)
    ap.add_argument("--contact-class-weight", nargs=2, type=float, default=None,
                     metavar=("NO_CONTACT", "CONTACT"),
                     help="Fixed (w_no_contact, w_contact) pair. Balanced data means this "
                          "should usually be left at the default (1.0, 1.0) -- unlike the "
                          "natural-ratio run, there's no severe imbalance left to correct for.")
    ap.add_argument("--contact-val-fraction", type=float, default=0.15,
                     help="Fraction of parent runs (positive bursts + raw negative runs) held "
                          "out for validation/early-stopping/threshold recalibration -- split "
                          "BEFORE any run is sub-chunked, to avoid near-duplicate frames from "
                          "the same physical stretch landing on both sides.")
    ap.add_argument("--contact-val-every-n-chunks", type=int, default=3)
    ap.add_argument("--contact-early-stop-patience", type=int, default=8,
                     help="Consecutive val checks with no improvement before stopping.")
    ap.add_argument("--contact-max-ratio", type=int, default=3,
                     help="Cap on same-class run/item skew, both in interleave_chunks' run "
                          "ordering and in the post-hoc per-chunk window-count rebalance.")
    ap.add_argument("--contact-norm-sample-items", type=int, default=20,
                     help="Number of TRAIN items decoded to compute the frozen global "
                          "mean/std passed to det.set_normalization().")
    ap.add_argument("--contact-min-precision", type=float, default=0.8,
                     help="Threshold/persistence-frames recalibration is restricted to "
                          "grid points with val precision >= this floor before ranking by "
                          "F1 -- same convention as calibrate_otsu_thresholds.py's "
                          "--min-precision, needed because an unconstrained F1 search on a "
                          "small validation set can pick a near-universal-positive "
                          "threshold that looks perfect on val but false-alarms constantly.")

    ap.add_argument("--skip-fire", action="store_true")
    ap.add_argument("--skip-human", action="store_true")
    ap.add_argument("--skip-contact", action="store_true")
    ap.add_argument("--only-contact", action="store_true", help="Shorthand for --skip-fire --skip-human.")
    args = ap.parse_args()
    if args.only_contact:
        args.skip_fire = True
        args.skip_human = True

    _setup_diagnostics()
    try:
        return _run(args)
    except BaseException:
        logger.exception("main() crashed")
        raise


def _run(args) -> int:
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise SystemExit(f"data root not found: {data_root}")

    waveshare_index = DatasetIndex(data_root / "waveshare_work", sensor_profile=WAVESHARE_26984, fps=8.0)
    fire_train_scenes, _ = build_task_split(waveshare_index.labeled_sessions(), task="fire")
    human_train_scenes, _ = build_task_split(waveshare_index.labeled_sessions(), task="human")
    contact_train_scenes, _ = build_task_split(waveshare_index.labeled_sessions(), task="contact")

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit([])
    registry = CheckpointRegistry(root=args.out)
    print(f"Checkpoints -> {registry.root}")

    if not args.skip_fire:
        train_fire_balanced(
            data_root, waveshare_index, fire_train_scenes, preprocessor, registry,
            synth_per_camera_samples=args.fire_synth_per_camera_samples,
            max_total=args.fire_max_total, seed=args.seed,
        )

    if not args.skip_human:
        train_human_balanced(
            waveshare_index, human_train_scenes, preprocessor, registry,
            exclude_scenes=set(args.exclude_human_scenes) or None,
            max_total=args.human_max_total, seed=args.seed, mobilenet_epochs=args.mobilenet_epochs,
        )

    if not args.skip_contact:
        class_weight = tuple(args.contact_class_weight) if args.contact_class_weight else None
        log_path = LOG_DIR / f"train_balanced_contact_losses_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        LOG_DIR.mkdir(exist_ok=True)
        print(f"Contact per-chunk loss log -> {log_path}")
        train_contact_balanced(
            data_root, waveshare_index, contact_train_scenes, preprocessor, registry,
            target_positive_frames=args.contact_target_positive_frames,
            target_negative_frames=args.contact_target_negative_frames,
            time_budget_hours=args.contact_time_budget_hours,
            checkpoint_every=args.contact_checkpoint_every,
            val_fraction=args.contact_val_fraction,
            val_every_n_chunks=args.contact_val_every_n_chunks,
            early_stop_patience=args.contact_early_stop_patience,
            max_ratio=args.contact_max_ratio,
            norm_sample_items=args.contact_norm_sample_items,
            min_precision=args.contact_min_precision,
            class_weight=class_weight, seed=args.seed, log_path=log_path,
        )

    print("\nAvailable checkpoints:")
    for algo, profile in registry.list_available():
        print(f"  {algo} / {profile or 'invariant'}")

    logger.info("main() completed normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
