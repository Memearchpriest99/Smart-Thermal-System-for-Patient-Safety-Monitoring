#!/usr/bin/env python3
"""Time-boxed ThermoX3D-only contact training on a stride-sampled subset of
the full corpus.

Why this exists (not just a rerun of train_full_corpus.py): the real
per-chunk rate for ThermoX3DDetector's chunked fit() over the full corpus
(~2072 waveshare_train + synth_room_1..5 chunks) is ~220s/chunk -- a ~5.4-day
job. Under a hard 12h deadline, that requires a ~10x reduction in total
compute. This script gets there by:

  1. Skipping FireSVMDetector/HOGSVMDetector/MobileNetSSDDetector/
     MVSTGCNDetector entirely -- their checkpoints already exist under
     checkpoints_full_corpus/ from the run that crashed partway through
     ThermoX3D (MVSTGCN's own chunked full-corpus fit had already completed
     and saved before the crash). Re-fitting them here would burn ~3h+ for
     no reason.
  2. Reusing the already-computed corpus-wide contact class_weight
     (no_contact=0.512, contact=21.164) instead of repeating the ~21min
     label-only counting pass -- it's a deterministic statistic of the same
     data/split, unchanged since the last run.
  3. Systematic stride sampling over the full ordered chunk stream (every
     `--stride`-th chunk) so the trained subset still touches every synth
     room + waveshare_train proportionally, rather than training on a
     contiguous prefix (which would mean waveshare_train + part of
     synth_room_1 only).
  4. Incremental checkpointing every `--checkpoint-every` accepted chunks
     (ThermoX3DDetector's own fit() loop only saves once, at the very end --
     a repeat crash would otherwise cost the whole run again, exactly what
     happened this morning).
  5. A hard wall-clock cutoff (`--time-budget-hours`) that stops the run
     even mid-subset, so this can never overrun the deadline regardless of
     rate variance.

LIMITATION to state in the report: this trains ThermoX3D on a ~1/stride
fraction of the corpus (default stride=11 -> ~9.1%), not the full corpus
like MVSTGCN got. This is a deliberate, documented tradeoff for a 12h
deadline, not a silent shortcut.
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
logger = logging.getLogger("train_thermox3d_subset")


def _setup_diagnostics() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    diag_path = LOG_DIR / f"train_thermox3d_subset_diag_{ts}.log"
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
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.training import ContactFrameDataset, DatasetIndex, Trainer  # noqa: E402
from scripts.train_full_corpus import (  # noqa: E402
    DATA_ROOT,
    DEFAULT_OUT,
    build_waveshare_split,
    contact_chunks,
)


class _PreprocessedDetector:
    """Mirrors eval_contact()'s inline wrapper in train_full_corpus.py --
    evaluate_contact_detection has no preprocessor hook of its own."""

    def __init__(self, inner, pre):
        self._inner = inner
        self._pre = pre

    def predict(self, frames):
        return self._inner.predict(tuple(self._pre.predict(f) for f in frames))

    def reset(self):
        if hasattr(self._inner, "reset"):
            self._inner.reset()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--contact-chunk-size", type=int, default=5000)
    ap.add_argument("--stride", type=int, default=11,
                     help="Keep every Nth chunk from the full ordered stream (systematic "
                          "sampling across all sessions). Default 11 -> ~9.1%% of the corpus.")
    ap.add_argument("--time-budget-hours", type=float, default=12.0,
                     help="Hard wall-clock cutoff -- stop even mid-subset if exceeded.")
    ap.add_argument("--checkpoint-every", type=int, default=15,
                     help="Save/overwrite the checkpoint every N *accepted* chunks.")
    ap.add_argument("--class-weight-no-contact", type=float, default=0.512,
                     help="Reused from the run that crashed 2026-08-08 -- deterministic "
                          "corpus statistic, same data/split, safe to reuse.")
    ap.add_argument("--class-weight-contact", type=float, default=21.164)
    ap.add_argument("--skip-eval", action="store_true")
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
    train_scenes, test_scenes = build_waveshare_split(waveshare_index, "contact")

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit([])
    registry = CheckpointRegistry(root=args.out)
    print(f"Checkpoints -> {registry.root}")

    class_weight = (args.class_weight_no_contact, args.class_weight_contact)
    print(f"  reusing class_weight = (no_contact={class_weight[0]:.3f}, contact={class_weight[1]:.3f}) "
          f"-- NOT recomputed this run, see script docstring")

    det = ThermoX3DDetector(sensor_profile=WAVESHARE_26984, class_weight=class_weight, n_epochs=1)

    time_budget_s = args.time_budget_hours * 3600.0
    t0 = time.time()
    n_seen = n_accepted = n_examples = 0
    stopped_reason = "subset exhausted"

    for source, frames_chunk, events_chunk in contact_chunks(
        data_root, waveshare_index, train_scenes, args.contact_chunk_size
    ):
        n_seen += 1
        if (n_seen - 1) % args.stride != 0:
            continue

        elapsed = time.time() - t0
        if elapsed > time_budget_s:
            stopped_reason = f"time budget exceeded ({elapsed:.0f}s > {time_budget_s:.0f}s)"
            break

        proc_chunk = [tuple(preprocessor.predict(f) for f in triplet) for triplet in frames_chunk]
        det.fit(proc_chunk, events_chunk)
        n_accepted += 1
        n_examples += len(frames_chunk)
        print(f"  chunk {n_accepted}/~{2072 // args.stride} (seen #{n_seen}, {source}): "
              f"{len(frames_chunk)} examples, {n_examples} total, "
              f"{time.time() - t0:.1f}s elapsed", flush=True)

        if n_accepted % args.checkpoint_every == 0:
            path = registry.register(det)
            print(f"  [checkpoint] saved after {n_accepted} accepted chunks -> {path}", flush=True)

    path = registry.register(det)
    print(f"\nThermoX3DDetector done ({stopped_reason}) -- {n_accepted} chunks / {n_examples} examples "
          f"-- saved -> {path}")

    if not args.skip_eval and test_scenes:
        print("\n--- thermo_x3d held-out eval (waveshare test scenes) ---")
        ds = ContactFrameDataset(waveshare_index, scenes=test_scenes)
        wrapped = _PreprocessedDetector(det, preprocessor)
        results = Trainer.evaluate_contact_detection(wrapped, ds, mode="full_corpus", verbose=False)
        total_correct = sum(r.confusion.correct for r in results)
        total_n = sum(r.confusion.total for r in results)
        print(f"  {len(results)} scenes, aggregate correct={total_correct}/{total_n}")
        for r in results:
            c = r.confusion
            print(f"    {r.name}: tp={c.tp} tn={c.tn} fp={c.fp} fn={c.fn}")

    logger.info("main() completed normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
