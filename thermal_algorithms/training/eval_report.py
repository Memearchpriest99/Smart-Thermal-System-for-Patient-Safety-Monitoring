"""Shared timed-evaluation harness for the full-corpus report (accuracy /
precision / recall / F1 / IoU + mean per-frame inference latency), used
identically across the fp32 baseline, ONNX, and quantized variants so every
number in the report comes from the same code path.

Unlike ``Trainer.evaluate_*`` (which returns one ``ScenarioResult`` per
session, meant for printing a per-scenario table), this module aggregates
every frame across the *entire* test split into one confusion matrix before
computing rates -- so a 3-frame session and a 3000-frame session are not
weighted equally, which per-session-mean-of-means would do.

Timing methodology: wall-clock ``time.perf_counter()`` around exactly one
``detector.predict(...)`` call per frame, single-frame batches throughout
(matches how the runtime pipeline actually calls detectors), with
``torch.cuda.synchronize()`` immediately before stopping the clock for any
torch-backed detector -- CUDA kernel launches are asynchronous, so omitting
the sync would time only host-side dispatch, not the actual GPU work. The
first N warm-up calls are discarded (CUDA context/cuDNN autotune warmup),
matching standard GPU-benchmarking practice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from thermal_algorithms.core.types import FireLevel
from thermal_algorithms.training.metrics import (
    BinaryConfusionMatrix,
    best_iou,
    binary_confusion_matrix,
)

WARMUP_CALLS = 10
"""Predict calls discarded from timing at the start of each evaluation (CUDA
context init / cuDNN autotune / first-call JIT are not representative of
steady-state per-frame latency)."""


_torch_usable: "Optional[bool]" = None


def _maybe_cuda_sync() -> None:
    """Best-effort torch.cuda.synchronize() -- no-op if torch or CUDA aren't
    available, so this module works unmodified for the sklearn detectors.

    Catches Exception broadly, not just ImportError: a blocked/missing DLL
    during torch's own import raises OSError (confirmed: a Windows
    Application Control policy actively blocking torch/lib/shm.dll did
    exactly this), which is not an ImportError subclass and would otherwise
    crash every detector's timed eval, including the pure-sklearn ones that
    never touch torch at all.

    Caches the outcome at module level after the first attempt: a *failed*
    import is not cached by Python itself, so without this, a persistently
    blocked torch would re-attempt (and re-fail) the same DLL load on every
    single frame -- confirmed to cost tens of milliseconds per attempt,
    which is both wasted work and, worse, silently inflates the very latency
    numbers this module exists to measure accurately for any torch-backed
    detector still able to import torch (this helper's own overhead would be
    counted inside their timed region).
    """
    global _torch_usable
    if _torch_usable is False:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _torch_usable = True
    except Exception:
        _torch_usable = False


@dataclass
class TimedEvalResult:
    """Aggregate result for one (detector, variant, test split) evaluation."""

    detector_name: str
    variant: str  # "fp32_baseline" | "onnx_fp32" | "fp16" | "bf16" | "int8" | ...
    task: str  # "fire" | "human" | "contact"
    confusion: BinaryConfusionMatrix
    mean_iou: Optional[float]  # None when IoU isn't applicable (contact)
    n_frames: int
    latencies_ms: list = field(default_factory=list, repr=False)
    # What the latencies above were actually measured on. Without this, a
    # latency number is uninterpretable -- 8 ms on a laptop GPU and 8 ms on
    # CPU say completely different things about deployability -- and the
    # report previously carried a hand-written claim about execution
    # providers that had drifted out of sync with the data.
    # ``device_source`` distinguishes a value captured at measurement time
    # ("measured") from one reconstructed afterwards from the deterministic
    # code path ("inferred"), so the two are never silently conflated.
    device: Optional[str] = None
    device_source: Optional[str] = None

    @property
    def accuracy(self) -> float:
        return self.confusion.accuracy

    @property
    def precision(self) -> float:
        return self.confusion.precision

    @property
    def recall(self) -> float:
        return self.confusion.recall

    @property
    def f1(self) -> float:
        return self.confusion.f1

    @property
    def mean_inference_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else float("nan")

    @property
    def p95_inference_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else float("nan")

    def to_dict(self) -> dict:
        return {
            "detector_name": self.detector_name,
            "variant": self.variant,
            "task": self.task,
            "n_frames": self.n_frames,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mean_iou": self.mean_iou,
            "mean_inference_ms": self.mean_inference_ms,
            "p95_inference_ms": self.p95_inference_ms,
            "device": self.device,
            "device_source": self.device_source,
            "tp": self.confusion.tp,
            "tn": self.confusion.tn,
            "fp": self.confusion.fp,
            "fn": self.confusion.fn,
        }


def describe_device(detector=None, *, onnx_providers=None, force: Optional[str] = None) -> str:
    """Human-readable description of what a measurement ran on.

    Resolution order:
      * ``force``           -- caller already knows (e.g. a shim it constructed);
      * ``onnx_providers``  -- ONNX Runtime: name the FIRST registered provider.
        Note this means "this provider was registered and takes priority", not
        "every node executed there" -- ORT partitions graphs and silently falls
        back to CPU per-node, so a stronger claim would need profiling;
      * a torch detector    -- report the concrete CUDA device name, because
        "cuda" alone tells a reader nothing about achievable latency;
      * otherwise           -- CPU (scikit-learn and the rule-based detectors
        have no GPU code path at all).
    """
    if force:
        return force
    if onnx_providers:
        first = onnx_providers[0] if isinstance(onnx_providers, (list, tuple)) else str(onnx_providers)
        if "CUDA" in first:
            return f"ONNX Runtime / {first} ({_cuda_name()})"
        return f"ONNX Runtime / {first}"
    dev = getattr(detector, "_device", None)
    if dev is None and hasattr(detector, "_device_str"):
        # A torch-backed detector whose model hasn't been built yet (or whose
        # ._model was swapped for a shim before _get_model() ran). Resolve the
        # SAME way the detector itself would, rather than falling through to
        # the CPU default below -- reporting "CPU" for a run that actually
        # used CUDA is precisely the mislabelling this column exists to stop.
        dev = getattr(detector, "_device_str", None) or (
            "cuda" if _cuda_available() else "cpu")
    if dev is not None and "cuda" in str(dev).lower():
        return f"{_cuda_name()} (CUDA)"
    if dev is not None:
        return str(dev).upper()
    return "CPU"


def _cuda_available() -> bool:
    if _torch_usable is False:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_name() -> str:
    # Only bail on a KNOWN-bad torch (False). `_torch_usable` is None until
    # the first _maybe_cuda_sync() call, and `not None` is True -- a plain
    # falsy check here would return the generic "CUDA" whenever this is
    # called before any timing has happened, which is exactly when the eval
    # scripts call it.
    if _torch_usable is False:
        return "CUDA"
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CUDA"


def _timed_predict(predict_fn: Callable[[], object], latencies_ms: list, warmup_left: list) -> object:
    """Call predict_fn() once, timing it unless still in the warmup window.

    warmup_left is a single-element mutable list used as an in/out counter
    so callers don't need to track warmup state themselves.
    """
    if warmup_left[0] > 0:
        warmup_left[0] -= 1
        return predict_fn()
    t0 = time.perf_counter()
    result = predict_fn()
    _maybe_cuda_sync()
    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    return result


def evaluate_fire_timed(
    detector,
    dataset,
    *,
    preprocessor=None,
    variant: str,
    detector_name: Optional[str] = None,
    device: Optional[str] = None,
    device_source: str = "measured",
) -> TimedEvalResult:
    y_true: list[int] = []
    y_pred: list[int] = []
    iou_scores: list[float] = []
    any_bbox_output = False
    latencies_ms: list[float] = []
    warmup_left = [WARMUP_CALLS]

    for session, examples in dataset.by_session():
        if hasattr(detector, "reset"):
            detector.reset()
        for frame, gt_alert in examples:
            if preprocessor is not None:
                frame = preprocessor.predict(frame)
            pred_alert = _timed_predict(lambda: detector.predict(frame), latencies_ms, warmup_left)

            gt_positive = gt_alert.level != FireLevel.SAFE
            pred_positive = pred_alert.level != FireLevel.SAFE
            y_true.append(1 if gt_positive else 0)
            y_pred.append(1 if pred_positive else 0)

            gt_bboxes = gt_alert.blob_features.get("bboxes", [])
            pred_bbox = pred_alert.blob_features.get("bbox")
            if pred_bbox is not None:
                any_bbox_output = True
            if gt_bboxes:
                pred_bboxes = [pred_bbox] if pred_bbox else []
                iou_scores.append(best_iou(pred_bboxes, gt_bboxes))

    cm = binary_confusion_matrix(y_true, y_pred)
    # None (not 0.0) when the detector's own output never carries a bbox at
    # all (e.g. FireSVMDetector's feature vector keeps scalar blob stats,
    # not spatial coordinates) -- that's "not applicable", distinct from a
    # bbox-producing detector (OtsuFireDetector) that just localizes poorly.
    if not any_bbox_output:
        mean_iou = None
    else:
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return TimedEvalResult(
        detector_name=detector_name or type(detector).__name__,
        variant=variant,
        task="fire",
        confusion=cm,
        mean_iou=mean_iou,
        n_frames=len(y_true),
        latencies_ms=latencies_ms,
        device=device or describe_device(detector),
        device_source=device_source,
    )


def evaluate_human_timed(
    detector,
    dataset,
    *,
    preprocessor=None,
    variant: str,
    detector_name: Optional[str] = None,
    device: Optional[str] = None,
    device_source: str = "measured",
) -> TimedEvalResult:
    y_true: list[int] = []
    y_pred: list[int] = []
    iou_scores: list[float] = []
    latencies_ms: list[float] = []
    warmup_left = [WARMUP_CALLS]

    for session, examples in dataset.by_session():
        if hasattr(detector, "reset"):
            detector.reset()
        for frame, gt_dets in examples:
            if preprocessor is not None:
                frame = preprocessor.predict(frame)
            pred_dets = _timed_predict(lambda: detector.predict(frame), latencies_ms, warmup_left)

            y_true.append(1 if gt_dets else 0)
            y_pred.append(1 if pred_dets else 0)

            gt_bboxes = [d.bbox for d in gt_dets]
            if gt_bboxes:
                pred_bboxes = [d.bbox for d in pred_dets if hasattr(d, "bbox")]
                iou_scores.append(best_iou(pred_bboxes, gt_bboxes))

    cm = binary_confusion_matrix(y_true, y_pred)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return TimedEvalResult(
        detector_name=detector_name or type(detector).__name__,
        variant=variant,
        task="human",
        confusion=cm,
        mean_iou=mean_iou,
        n_frames=len(y_true),
        latencies_ms=latencies_ms,
        device=device or describe_device(detector),
        device_source=device_source,
    )


def evaluate_contact_timed(
    detector,
    dataset,
    *,
    variant: str,
    detector_name: Optional[str] = None,
    device: Optional[str] = None,
    device_source: str = "measured",
) -> TimedEvalResult:
    y_true: list[int] = []
    y_pred: list[int] = []
    latencies_ms: list[float] = []
    warmup_left = [WARMUP_CALLS]

    for session, examples in dataset.by_session():
        if hasattr(detector, "reset"):
            detector.reset()
        for frames, gt_event in examples:
            pred_event = _timed_predict(lambda: detector.predict(frames), latencies_ms, warmup_left)
            y_true.append(1 if gt_event.any_contact else 0)
            y_pred.append(1 if pred_event.any_contact else 0)

    cm = binary_confusion_matrix(y_true, y_pred)
    return TimedEvalResult(
        detector_name=detector_name or type(detector).__name__,
        variant=variant,
        task="contact",
        confusion=cm,
        mean_iou=None,
        n_frames=len(y_true),
        latencies_ms=latencies_ms,
        device=device or describe_device(detector),
        device_source=device_source,
    )
