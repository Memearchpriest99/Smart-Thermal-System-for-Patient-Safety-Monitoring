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
Contact: NOT handled here any more. Contact training moved to
         scripts/train_contact_real_optuna.py (real data only + augmentation
         + Optuna) when the project owner dropped synthetic data from the
         contact pipeline on 2026-08-09. The synth-based chunked loop that
         used to live in this file produced the S4 ThermoX3D numbers in
         reports/Full_Corpus_Engineering_Report.pdf and is recoverable from git history.

All checkpoints register to a NEW root (--out, default checkpoints_balanced/)
-- never the same root as checkpoints_full_corpus/, so this can never
overwrite the natural-ratio weights the existing report is built on.

Usage::

    python scripts/train_balanced_corpus.py              # fire + human
    python scripts/train_balanced_corpus.py --skip-fire  # human only
"""
from __future__ import annotations

import argparse
import faulthandler
import logging
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
from thermal_algorithms.training import DatasetIndex, build_task_split  # noqa: E402
from thermal_algorithms.training.balance import (  # noqa: E402
    build_balanced_fire_pool,
    build_balanced_human_pool,
)

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

    ap.add_argument("--skip-fire", action="store_true")
    ap.add_argument("--skip-human", action="store_true")
    args = ap.parse_args()

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

    print("\nContact detection is trained separately -- see "
          "scripts/train_contact_real_optuna.py.")

    print("\nAvailable checkpoints:")
    for algo, profile in registry.list_available():
        print(f"  {algo} / {profile or 'invariant'}")

    logger.info("main() completed normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
