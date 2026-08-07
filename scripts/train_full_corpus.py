#!/usr/bin/env python3
"""Task 3 driver: re-train every trainable detector on the full corpus
(synth_room_1..5 + waveshare_work), using ``GlobalNormPreprocessor`` — the
winning pipeline from the Task 2 comparison
(reports/preprocessing_comparison_results.json: better SBR, fire accuracy/
recall, and contact accuracy than TatenoPipeline; human detection ~tied).

Scope (per data/DATASET_NOTES.md):
  - room-1 is excluded entirely (corrupted; both recording dates unusable).
  - Synthetic data has no bounding boxes, ever, so HOGSVMDetector /
    MobileNetSSDDetector train on waveshare_work only.
  - FireSVMDetector and the contact detectors (MVSTGCNDetector,
    ThermoX3DDetector) train on the full corpus: waveshare_work's TRAIN
    scenes + every synth_room session (synthetic data is used wholesale,
    never split — per the task-3 instruction).
  - waveshare_work is split by whole session (seeded, ~14 train / 3 test)
    so every trainable detector has a held-out sanity-check eval.

Contact detectors (MV-STGCN, Thermo-X3D) build sliding-window training
examples by materializing their `fit(X, y)` args into a list internally, so
the full corpus can never be handed to them in one call (a single synthetic
session alone can be ~1.45M timesteps). `iter_contact_training_chunks`
solves this by yielding bounded, session-contiguous chunks; this script
calls `fit()` once per chunk, continuing training of the same model
instance each time (weights persist across `fit()` calls; only the
optimizer and this chunk's global-normalization stats are rebuilt each
call — see the class docstrings).

`class_weight` for MVSTGCNDetector/ThermoX3DDetector must be a fixed
``(w_no_contact, w_contact)`` pair supplied at construction (unlike
sklearn's ``'balanced'`` string, it can't be computed lazily inside `fit()`
since the same instance is reused across chunks). This script runs one
label-only counting pass over the corpus first to compute it -- frame data
is still read from disk during that pass (no cheaper option, since labels
are paired with frames at the source), but the expensive per-frame
preprocessing is skipped, so it's far cheaper than the real training pass.

Usage::

    python scripts/train_full_corpus.py
    python scripts/train_full_corpus.py --skip-contact       # fire+human only
    python scripts/train_full_corpus.py --skip-fire --skip-human
    python scripts/train_full_corpus.py --contact-chunk-size 2000
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import signal
import sys
import time
from itertools import tee
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG_DIR = _REPO_ROOT / "logs"
logger = logging.getLogger("train_full_corpus")


def _setup_diagnostics() -> Path:
    """Best-effort diagnostics for a run that dies with no Python traceback
    (native crash in a C extension, or an external kill signal) -- both leave
    stdout/the normal try/except silent, so this writes to its own file:

    - faulthandler.enable(): installs a fatal-signal / Windows-fatal-exception
      handler that dumps a full C-level traceback on crashes that never reach
      normal Python exception handling (e.g. a segfault inside cv2/torch).
      Passive -- only fires on an actual fatal signal, so it can't itself
      perturb a healthy run.
    - signal handlers: log receipt of SIGINT/SIGTERM/SIGBREAK before exiting,
      in case this is a graceful external stop rather than a crash.

    Deliberately NOT using faulthandler.dump_traceback_later(): its periodic
    watchdog thread walks every thread's C stack on a timer regardless of
    what the main thread is doing, and empirically that collided with cv2
    native calls (Windows fatal exception: access violation, reproducibly at
    the ~60s mark -- exactly the configured dump interval) that did not occur
    at all with faulthandler.enable() alone. The cure was worse than the
    disease; don't reintroduce it without confirming this platform/opencv
    combination is actually safe with it.
    """
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    diag_path = LOG_DIR / f"train_full_corpus_diag_{ts}.log"
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

    logger.info("diagnostics active")
    print(f"Diagnostics (crash/signal/hang traces) -> {diag_path}", flush=True)
    return diag_path


def _log_every(iterable, label: str, every: int = 500):
    """Pass-through generator wrapper that logs a heartbeat every `every`
    items, so a run that dies mid-stream still tells us how far it got."""
    t0 = time.time()
    n = 0
    for item in iterable:
        n += 1
        if n % every == 0:
            logger.info(f"{label}: {n} items, {time.time() - t0:.1f}s elapsed")
        yield item
    logger.info(f"{label}: done, {n} items total, {time.time() - t0:.1f}s elapsed")

from thermal_algorithms.core.checkpoints import CheckpointRegistry  # noqa: E402
from thermal_algorithms.core.sensor_profile import WAVESHARE_26984  # noqa: E402
from thermal_algorithms.preprocessing.global_norm_pipeline import GlobalNormPreprocessor  # noqa: E402
from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector  # noqa: E402
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector  # noqa: E402
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector  # noqa: E402
from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector  # noqa: E402
from thermal_algorithms.training import (  # noqa: E402
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    PERSON_CLASS_ID,
    Trainer,
    format_scenario_table,
    iter_synth_sessions,
    session_train_test_split,
)

try:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

DATA_ROOT = _REPO_ROOT.parent / "data"
DEFAULT_OUT = _REPO_ROOT / "checkpoints_full_corpus"


# ---------------------------------------------------------------------------
# Waveshare session split
# ---------------------------------------------------------------------------

def build_waveshare_split(waveshare_index: DatasetIndex) -> tuple[set[str], set[str]]:
    train_sessions, test_sessions = session_train_test_split(waveshare_index.labeled_sessions())
    train_scenes = {s.scene for s in train_sessions}
    test_scenes = {s.scene for s in test_sessions}
    print(f"waveshare split: {len(train_scenes)} train scenes / {len(test_scenes)} test scenes")
    if not test_scenes:
        print("  WARNING: no held-out waveshare scenes -- eval step will be skipped.")
    return train_scenes, test_scenes


# ---------------------------------------------------------------------------
# Fire
# ---------------------------------------------------------------------------

def train_fire(data_root: Path, waveshare_index: DatasetIndex, train_scenes: set[str],
                preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry) -> FireSVMDetector:
    print("\n=== Training FireSVMDetector on the full corpus ===")

    def examples():
        yield from FireFrameDataset(waveshare_index, scenes=train_scenes)
        for session in iter_synth_sessions(data_root):
            yield from session.fire_examples()

    ex_for_x, ex_for_y = tee(examples(), 2)
    ex_for_x = _log_every(ex_for_x, "fire-train frames")
    X = (preprocessor.predict(frame) for frame, _ in ex_for_x)
    y = (alert for _, alert in ex_for_y)

    det = FireSVMDetector(sensor_profile=WAVESHARE_26984, class_weight="balanced")
    t0 = time.time()
    logger.info("FireSVMDetector.fit() starting")
    det.fit(X, y)
    logger.info("FireSVMDetector.fit() returned")
    print(f"  fit() done in {time.time() - t0:.1f}s")
    print(f"  saved -> {registry.register(det)}")
    return det


def eval_fire(det: FireSVMDetector, waveshare_index: DatasetIndex, test_scenes: set[str],
              preprocessor: GlobalNormPreprocessor) -> None:
    if not test_scenes:
        return
    print("\n--- FireSVMDetector held-out eval (waveshare test scenes) ---")
    ds = FireFrameDataset(waveshare_index, scenes=test_scenes)
    results = Trainer.evaluate_fire_detection(det, ds, preprocessor=preprocessor, mode="full_corpus", verbose=False)
    print(format_scenario_table(results, include_iou=True))


# ---------------------------------------------------------------------------
# Human (waveshare only -- synth has no bounding boxes, ever)
# ---------------------------------------------------------------------------

def train_human(waveshare_index: DatasetIndex, train_scenes: set[str],
                 preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry) -> dict:
    print("\n=== Training human detectors on waveshare_work (train scenes only) ===")
    ds = FrameLevelDataset(waveshare_index, scenes=train_scenes, class_filter=[PERSON_CLASS_ID])
    examples = [(preprocessor.predict(frame), dets) for frame, dets in ds]
    print(f"  {len(examples)} labeled waveshare frames")

    detectors = {}

    hog = HOGSVMDetector(sensor_profile=WAVESHARE_26984, class_weight="balanced")
    t0 = time.time()
    hog.fit(examples)
    print(f"  HOGSVMDetector fit() done in {time.time() - t0:.1f}s -- saved -> {registry.register(hog)}")
    detectors["hog_svm"] = hog

    if _TORCH_OK:
        mnet = MobileNetSSDDetector(sensor_profile=WAVESHARE_26984)
        t0 = time.time()
        mnet.fit(examples)
        print(f"  MobileNetSSDDetector fit() done in {time.time() - t0:.1f}s -- saved -> {registry.register(mnet)}")
        detectors["mobilenet_ssd"] = mnet
    else:
        print("  torch not available -- skipping MobileNetSSDDetector")

    return detectors


def eval_human(detectors: dict, waveshare_index: DatasetIndex, test_scenes: set[str],
               preprocessor: GlobalNormPreprocessor) -> None:
    if not test_scenes:
        return
    ds = FrameLevelDataset(
        waveshare_index, scenes=test_scenes, class_filter=[PERSON_CLASS_ID], include_negative_frames=True,
    )
    for label, det in detectors.items():
        print(f"\n--- {label} held-out eval (waveshare test scenes) ---")
        results = Trainer.evaluate_human_detection(det, ds, preprocessor=preprocessor, mode="full_corpus", verbose=False)
        print(format_scenario_table(results, include_iou=True))


# ---------------------------------------------------------------------------
# Contact -- chunked training over the full corpus
# ---------------------------------------------------------------------------

def _chunks(iterable, source_label: str, chunk_size: int):
    buf_frames, buf_events = [], []
    for frames, event in iterable:
        buf_frames.append(frames)
        buf_events.append(event)
        if len(buf_frames) >= chunk_size:
            yield source_label, buf_frames, buf_events
            buf_frames, buf_events = [], []
    if buf_frames:
        yield source_label, buf_frames, buf_events


def contact_chunks(data_root: Path, waveshare_index: DatasetIndex, train_scenes: set[str], chunk_size: int):
    """Train-scene-filtered waveshare, then every synth session -- mirrors
    `full_corpus.iter_contact_training_chunks` but scoped to the waveshare
    TRAIN split rather than the whole index."""
    yield from _chunks(ContactFrameDataset(waveshare_index, scenes=train_scenes), "waveshare_train", chunk_size)
    for session in iter_synth_sessions(data_root):
        yield from _chunks(session.contact_examples(), session.scene, chunk_size)


def compute_contact_class_weight(data_root: Path, waveshare_index: DatasetIndex,
                                  train_scenes: set[str]) -> tuple[float, float]:
    print("  counting contact labels across the corpus (label-only pass, no preprocessing)...")
    n_pos = n_neg = 0
    t0 = time.time()

    def label_examples():
        yield from ContactFrameDataset(waveshare_index, scenes=train_scenes)
        for session in iter_synth_sessions(data_root):
            yield from session.contact_examples()

    for _frames, event in label_examples():
        if event.any_contact:
            n_pos += 1
        else:
            n_neg += 1
    total = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: degenerate label counts (pos={n_pos}, neg={n_neg}) -- falling back to unweighted.")
        return (1.0, 1.0)
    w_pos = total / (2.0 * n_pos)
    w_neg = total / (2.0 * n_neg)
    print(f"  contact={n_pos} ({n_pos/total:.1%})  no-contact={n_neg} ({n_neg/total:.1%})  "
          f"[{time.time() - t0:.1f}s]")
    print(f"  class_weight = (no_contact={w_neg:.3f}, contact={w_pos:.3f})")
    return (w_neg, w_pos)


def train_contact(data_root: Path, waveshare_index: DatasetIndex, train_scenes: set[str],
                   preprocessor: GlobalNormPreprocessor, registry: CheckpointRegistry,
                   chunk_size: int) -> dict:
    print("\n=== Training contact detectors on the full corpus (chunked) ===")
    class_weight = compute_contact_class_weight(data_root, waveshare_index, train_scenes)

    detectors = {}
    for label, cls in (("mv_stgcn", MVSTGCNDetector), ("thermo_x3d", ThermoX3DDetector)):
        print(f"\n--- {cls.__name__} ---")
        det = cls(sensor_profile=WAVESHARE_26984, class_weight=class_weight)
        t0 = time.time()
        n_chunks = n_examples = 0
        for source, frames_chunk, events_chunk in contact_chunks(data_root, waveshare_index, train_scenes, chunk_size):
            proc_chunk = [tuple(preprocessor.predict(f) for f in triplet) for triplet in frames_chunk]
            det.fit(proc_chunk, events_chunk)
            n_chunks += 1
            n_examples += len(frames_chunk)
            print(f"  chunk {n_chunks} ({source}): {len(frames_chunk)} examples, "
                  f"{n_examples} total, {time.time() - t0:.1f}s elapsed", flush=True)
        print(f"  {cls.__name__} done -- saved -> {registry.register(det)}")
        detectors[label] = det

    return detectors


def eval_contact(detectors: dict, waveshare_index: DatasetIndex, test_scenes: set[str],
                  preprocessor: GlobalNormPreprocessor) -> None:
    if not test_scenes:
        return

    class _PreprocessedDetector:
        """evaluate_contact_detection has no preprocessor hook of its own
        (unlike evaluate_fire_detection/evaluate_human_detection) -- wrap
        the detector instead, matching scripts/eval_preprocessing_comparison.py."""

        def __init__(self, inner, pre):
            self._inner = inner
            self._pre = pre

        def predict(self, frames):
            return self._inner.predict(tuple(self._pre.predict(f) for f in frames))

        def reset(self):
            if hasattr(self._inner, "reset"):
                self._inner.reset()

    ds = ContactFrameDataset(waveshare_index, scenes=test_scenes)
    for label, det in detectors.items():
        print(f"\n--- {label} held-out eval (waveshare test scenes) ---")
        wrapped = _PreprocessedDetector(det, preprocessor)
        results = Trainer.evaluate_contact_detection(wrapped, ds, mode="full_corpus", verbose=False)
        total_correct = sum(r.confusion.correct for r in results)
        total_n = sum(r.confusion.total for r in results)
        print(f"  {len(results)} scenes, aggregate correct={total_correct}/{total_n}")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--contact-chunk-size", type=int, default=5000)
    ap.add_argument("--skip-fire", action="store_true")
    ap.add_argument("--skip-human", action="store_true")
    ap.add_argument("--skip-contact", action="store_true")
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
    train_scenes, test_scenes = build_waveshare_split(waveshare_index)

    preprocessor = GlobalNormPreprocessor(WAVESHARE_26984)
    preprocessor.fit()

    registry = CheckpointRegistry(root=args.out)
    print(f"Checkpoints -> {registry.root}")

    if not args.skip_fire:
        logger.info("stage: fire")
        det = train_fire(data_root, waveshare_index, train_scenes, preprocessor, registry)
        if not args.skip_eval:
            eval_fire(det, waveshare_index, test_scenes, preprocessor)

    if not args.skip_human:
        logger.info("stage: human")
        detectors = train_human(waveshare_index, train_scenes, preprocessor, registry)
        if not args.skip_eval:
            eval_human(detectors, waveshare_index, test_scenes, preprocessor)

    if not args.skip_contact:
        logger.info("stage: contact")
        detectors = train_contact(data_root, waveshare_index, train_scenes, preprocessor, registry, args.contact_chunk_size)
        if not args.skip_eval:
            eval_contact(detectors, waveshare_index, test_scenes, preprocessor)

    print("\nAvailable checkpoints:")
    for algo, profile in registry.list_available():
        print(f"  {algo} / {profile or 'invariant'}")

    logger.info("main() completed normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
