"""Drop-in ONNX Runtime replacements for each detector's learned
sub-component (see scripts/export_onnx_models.py's module docstring for
exactly what is/isn't traced into each ONNX graph).

Each shim exposes the same call interface the surrounding detector code
already uses -- `.predict()`/`.predict_proba()`/`.decision_function()` for
the sklearn detectors, `__call__(*tensors) -> torch.Tensor(s)` for the torch
ones -- so swapping `detector._svm` / `detector._scaler` / `detector._model`
for a shim instance is the ONLY change needed to benchmark the ONNX path;
every line of pre/post-processing in the detector's own predict() runs
completely unmodified. This is what makes the fp32-vs-ONNX comparison
apples-to-apples.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import onnxruntime as ort


def make_session(onnx_path, prefer_gpu: bool = True) -> tuple[ort.InferenceSession, list[str]]:
    """Create an ONNX Runtime session, preferring CUDA if it actually
    initializes. Listing CUDAExecutionProvider as "available" doesn't
    guarantee its native deps load cleanly for this CUDA/driver combination
    -- fall back to CPU on any init failure rather than crashing, and report
    which providers actually ended up active (session.get_providers()) so
    the report can state honestly which one ran."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if prefer_gpu else ["CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess, sess.get_providers()


class _Identity:
    """Replaces a StandardScaler when its transform is already baked into
    the exported ONNX graph -- avoids double-applying it."""

    def transform(self, X):
        return X


class OnnxFireSvmShim:
    """Replaces FireSVMDetector._svm (use alongside detector._scaler =
    _Identity() so the combined scaler+SVC ONNX graph runs exactly once)."""

    def __init__(self, onnx_path, prefer_gpu: bool = True) -> None:
        self.session, self.providers = make_session(onnx_path, prefer_gpu)
        self._input_name = self.session.get_inputs()[0].name
        self.classes_ = np.array([0, 1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        label, _ = self.session.run(None, {self._input_name: np.asarray(X, dtype=np.float32)})
        return np.asarray(label)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, proba = self.session.run(None, {self._input_name: np.asarray(X, dtype=np.float32)})
        return np.asarray(proba)


class OnnxLinearSvcShim:
    """Replaces HOGSVMDetector._svm. decision_function(X) returns the raw
    margin. skl2onnx names LinearSVC's raw-score output "probabilities" as a
    naming artifact of the converter -- LinearSVC has no predict_proba; this
    is the same quantity sklearn's own .decision_function() returns."""

    def __init__(self, onnx_path, prefer_gpu: bool = True) -> None:
        self.session, self.providers = make_session(onnx_path, prefer_gpu)
        self._input_name = self.session.get_inputs()[0].name

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        _, scores = self.session.run(None, {self._input_name: np.asarray(X, dtype=np.float32)})
        scores = np.asarray(scores)
        return scores[:, 0] if scores.ndim == 2 else scores.ravel()


class OnnxTorchShim:
    """Generic replacement for a torch nn.Module: consumes/produces torch
    tensors so surrounding code that calls `model(t)` and then runs
    F.softmax / .item() / indexing on the result needs no changes.

    extra_attrs lets callers preserve non-tensor state the original module
    exposed that downstream code still reads directly (e.g.
    MobileNetSSDDetector.predict() reads `model.anchors`, which is a fixed
    property of the original module, not an ONNX graph output).
    """

    def __init__(
        self,
        onnx_path,
        input_names: list[str],
        prefer_gpu: bool = True,
        **extra_attrs,
    ) -> None:
        self.session, self.providers = make_session(onnx_path, prefer_gpu)
        self._input_names = input_names
        for k, v in extra_attrs.items():
            setattr(self, k, v)

    def __call__(self, *tensors):
        import torch

        feed = {
            name: t.detach().cpu().numpy().astype(np.float32)
            for name, t in zip(self._input_names, tensors)
        }
        outputs = self.session.run(None, feed)
        torch_outputs = [torch.from_numpy(o) for o in outputs]
        return torch_outputs[0] if len(torch_outputs) == 1 else tuple(torch_outputs)

    def eval(self):
        return self

    def to(self, device):
        return self
