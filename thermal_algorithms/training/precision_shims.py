"""Precision-cast wrapper for benchmarking fp16/bf16 inference against the
fp32 baseline while holding every other line of detector code identical --
same drop-in-shim spirit as onnx_shims.py, just swapping dtype instead of
framework.

Only meaningful for the three torch-backed detectors (MobileNetSSDDetector,
MVSTGCNDetector, ThermoX3DDetector). scikit-learn's SVC/LinearSVC always run
their C-implemented (libsvm/liblinear) decision function in float64 on CPU --
there is no mixed-precision code path to engage and no GPU tensor-core
benefit to gain, so casting their inputs to fp16 would just add a
meaningless cast with no different arithmetic underneath. That's a real
finding, not a gap: it's why scripts/eval_quantized_models.py only exercises
int8 (via ONNX Runtime dynamic quantization) for the sklearn detectors.
"""

from __future__ import annotations


class CastModelShim:
    """Wraps a torch nn.Module: runs forward in `dtype`, casts output(s)
    back to float32 so unmodified downstream code (anchor decoding,
    F.softmax, .item()) sees the same tensor dtype it always has -- isolates
    "does computing in fp16/bf16 change the result" from any other effect.
    Model and inputs stay on whatever device they're already on (CUDA, for
    the default-constructed detectors) -- only dtype changes, since the
    fp16/bf16 tensor-core speedup is GPU-only.
    """

    def __init__(self, model, dtype, **extra_attrs) -> None:
        self._model = model.to(dtype)
        self._dtype = dtype
        for k, v in extra_attrs.items():
            setattr(self, k, v)

    def __call__(self, *tensors):
        casted = [t.to(self._dtype) for t in tensors]
        out = self._model(*casted)
        if isinstance(out, tuple):
            return tuple(o.float() for o in out)
        return out.float()

    def eval(self):
        self._model.eval()
        return self

    def to(self, device):
        self._model = self._model.to(device)
        return self
