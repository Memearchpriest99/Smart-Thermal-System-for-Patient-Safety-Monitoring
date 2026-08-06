"""GlobalNormPreprocessor — frame-level normalization baseline, no learned
background. Built for the Task 2 preprocessing comparison against
`TatenoPipeline`.

Redesigned from the annotation tool's client-side display filter
(`annontation_tool/static/index.html:2382-2530`) — a Gaussian blur followed by
subtracting the *current frame's own* mean luminance, meant as a display aid
for human annotators, operating on post-colormap 8-bit RGB with no
calibration step at all. That filter isn't a genuine background-subtraction
algorithm (its "background" is just the same frame's own scalar mean, not a
learned spatial map), so it isn't ported literally — this class captures the
same underlying idea (background estimation with no learned/calibrated
state — only ever the current frame) as a real `Preprocessor` operating on
raw Celsius `Frame` arrays, matching `TatenoPipeline`'s 3-stage shape so the
two are directly comparable via `Trainer.evaluate_preprocessing`.

Mathematics (deliberately parallel to `TatenoPipeline`'s notation)
-------------------------------------------------------------------
    Smoothing:   I_s(x,y,t) = (G_{sigma,sigma} * I)(x,y,t)     [identical to Tateno]
    Background:  B_t = mean_{x,y} I_s(x,y,t)                   <- per-frame SCALAR,
                                                                    recomputed every call
    Difference:  D(x,y,t) = I_s(x,y,t) - B_t
    Output:      Output(x,y,t) = |D(x,y,t)|                    [identical L1 residual]

The one deliberate difference from `TatenoPipeline` is the background term:
Tateno's B(x,y) is a per-pixel spatial map learned once from empty-room
calibration frames (`fit()`); this class's B_t is a single scalar recomputed
fresh from every incoming frame, with no calibration state at all. Sigma/
kernel-size resolution is intentionally identical to `TatenoPipeline`'s (same
physical-smoothing-scale convention) so the comparison isolates that one
difference rather than being confounded by different blur.
"""

from __future__ import annotations

from math import ceil
from typing import Iterable, Optional, Union

import cv2
import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing.base import Preprocessor


class GlobalNormPreprocessor(Preprocessor):
    """Gaussian smooth -> subtract the frame's own mean -> L1 residual.

    No learned background: `fit()` is a genuine no-op (nothing to calibrate),
    present only to satisfy the `Preprocessor` contract and so calling code
    that fits every preprocessor uniformly doesn't need to special-case this
    one.
    """

    name = "global_norm_pipeline"
    is_trainable = False
    resolution_behavior = "parameterized"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        sigma: Optional[float] = None,
        physical_smoothing_m: float = 0.05,
        assumed_distance_m: float = 2.0,
        kernel_size: Optional[int] = None,
    ) -> None:
        """See `TatenoPipeline.__init__` — the smoothing parameters mean
        exactly the same thing here, resolved the same way, for a fair
        comparison."""
        if sensor_profile is None:
            raise ValueError("GlobalNormPreprocessor requires a SensorProfile.")

        super().__init__(
            sensor_profile=sensor_profile,
            sigma=sigma,
            physical_smoothing_m=physical_smoothing_m,
            assumed_distance_m=assumed_distance_m,
            kernel_size=kernel_size,
        )

        self._sigma: float = self._resolve_sigma(
            sensor_profile, sigma, physical_smoothing_m, assumed_distance_m
        )
        self._kernel_size: int = self._resolve_kernel_size(kernel_size, self._sigma)

    # ---- Internal resolvers (identical convention to TatenoPipeline) ------

    @staticmethod
    def _resolve_sigma(
        profile: SensorProfile,
        sigma: Optional[float],
        physical_smoothing_m: float,
        assumed_distance_m: float,
    ) -> float:
        if sigma is not None:
            if sigma <= 0:
                raise ValueError(f"sigma must be positive; got {sigma}.")
            return float(sigma)
        if physical_smoothing_m <= 0 or assumed_distance_m <= 0:
            raise ValueError(
                "physical_smoothing_m and assumed_distance_m must be positive."
            )
        dx_m, dy_m = profile.physical_pixel_size_m(assumed_distance_m)
        avg_pixel_m = (dx_m + dy_m) / 2.0
        return float(physical_smoothing_m / avg_pixel_m)

    @staticmethod
    def _resolve_kernel_size(kernel_size: Optional[int], sigma: float) -> int:
        if kernel_size is None:
            kernel_size = 2 * int(ceil(3 * sigma)) + 1
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}.")
        return int(kernel_size)

    # ---- Read-only resolved parameters -----------------------------------

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    # ---- Internal building blocks ------------------------------------------

    def _smooth(self, frame_data: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            frame_data.astype(np.float32),
            ksize=(self._kernel_size, self._kernel_size),
            sigmaX=self._sigma,
            sigmaY=self._sigma,
            borderType=cv2.BORDER_REFLECT,
        )

    def _validate_shape(self, data: np.ndarray) -> None:
        ew, eh = self.sensor_profile.resolution  # type: ignore[union-attr]
        if data.shape != (eh, ew):
            raise ValueError(
                f"Frame shape {data.shape} does not match "
                f"{self.sensor_profile.name} expected (H, W) = ({eh}, {ew})."  # type: ignore[union-attr]
            )

    # ---- Public API --------------------------------------------------------

    def fit(
        self,
        X: Union[Iterable[Frame], np.ndarray, None] = None,
        y: None = None,
    ) -> "GlobalNormPreprocessor":
        """No-op — there is no background to learn. Accepts (and ignores) an
        optional calibration set purely so code that calls `fit()` uniformly
        across preprocessors (e.g. on empty-room frames, for `TatenoPipeline`
        in the same loop) doesn't need to special-case this class."""
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> Frame:
        """Apply the pipeline to one frame.

        Returns a new `Frame` whose `data` is the rectified residual;
        timestamp and camera_id are preserved. Unlike `TatenoPipeline`, this
        never raises for "not fitted" in a meaningful sense (there's nothing
        to have learned) — the `is_fitted` gate is still enforced for
        workflow consistency with `TatenoPipeline`.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "GlobalNormPreprocessor.predict() called before fit(). Call "
                "fit() first (even though it's a no-op) for workflow "
                "consistency with other Preprocessors."
            )
        self._validate_shape(X.data)

        smoothed = self._smooth(X.data)
        frame_mean = float(smoothed.mean())
        diff = smoothed - frame_mean
        residual = np.abs(diff).astype(np.float32)

        return Frame(
            data=residual,
            timestamp=X.timestamp,
            camera_id=X.camera_id,
            metadata={**X.metadata, "preprocessed_by": self.name, "frame_mean_c": frame_mean},
        )

    # ---- Persistence -------------------------------------------------------

    def _state_dict(self) -> dict:
        return {
            "resolved_sigma": self._sigma,
            "resolved_kernel_size": self._kernel_size,
        }

    def _load_state_dict(self, state: dict) -> None:
        if "resolved_sigma" in state:
            self._sigma = float(state["resolved_sigma"])
        if "resolved_kernel_size" in state:
            self._kernel_size = int(state["resolved_kernel_size"])
        self._is_fitted = True
