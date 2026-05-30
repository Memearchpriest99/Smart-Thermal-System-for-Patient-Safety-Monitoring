"""Tateno pre-processing pipeline — Engineering Report § 4.4.1.

Three-stage transformation applied to every incoming thermal frame:

    1. Spatial Denoising         — 2-D Gaussian smoothing
    2. Background Subtraction    — mean-field environmental filtering
    3. Residual Rectification    — L1 magnitude

The pipeline turns a raw thermal frame into a clean residual where the
remaining signal energy corresponds to scene anomalies (humans, fires)
rather than static environmental emissions (radiators, walls, electronics).

Mathematics (matching the report's notation)
---------------------------------------------
    Smoothing:   I_s(x,y,t) = (G_{σ,σ} * I)(x,y,t)
    Background:  B(x,y) = (1/K) Σ_{k=1..K} I_s(·,·,k)        ← learned in fit()
    Difference:  D(x,y,t) = I_s(x,y,t) - B(x,y)
    Output:      Output(x,y,t) = |D(x,y,t)|                   ← L1 (Manhattan) norm

Note that B is computed over the *smoothed* calibration frames. The report's
formula is over raw I_calib, but smoothing the calibration set the same way
the inference frames are smoothed keeps the subtraction consistent and
suppresses high-frequency calibration noise that would otherwise survive
into the residual.

Resolution scaling
------------------
This class is `parameterized` over the SensorProfile. The Gaussian σ defaults
to a *physical* smoothing scale (in meters, default 5 cm) which is converted
to pixels at a configurable assumed working distance, giving the same
effective spatial smoothing on both the 32×24 MLX and 80×62 Waveshare paths.
The kernel size is auto-derived from σ (≈ 6σ+1, rounded to the nearest odd ≥ 3).
"""

from __future__ import annotations

from math import ceil
from typing import Iterable, Optional, Union

import cv2
import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Frame
from thermal_algorithms.preprocessing.base import Preprocessor


class TatenoPipeline(Preprocessor):
    """Reference implementation of the § 4.4.1 pre-processing pipeline."""

    name = "tateno_pipeline"
    is_trainable = False
    resolution_behavior = "parameterized"

    # ---- Construction -----------------------------------------------------

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        sigma: Optional[float] = None,
        physical_smoothing_m: float = 0.05,
        assumed_distance_m: float = 2.0,
        kernel_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            sensor_profile: REQUIRED. Sets the expected frame shape and is
                used to convert the physical smoothing scale to pixel σ.
            sigma: Gaussian standard deviation in *pixels*. If None (default),
                derived from `physical_smoothing_m` and `assumed_distance_m`.
            physical_smoothing_m: Real-world smoothing scale in meters. The
                default 5 cm is conservative — it preserves human-scale features
                while attenuating per-pixel noise. Used only when `sigma` is None.
            assumed_distance_m: Working range used to convert physical units
                to pixels. The default 2.0 m matches the project's typical
                room-monitoring geometry. Used only when `sigma` is None.
            kernel_size: Gaussian kernel size in pixels (must be odd ≥ 3). If
                None (default), derived as `2 * ceil(3 * sigma) + 1`.
        """
        if sensor_profile is None:
            raise ValueError("TatenoPipeline requires a SensorProfile.")

        # Persist user-facing hyperparameters so get_params() round-trips.
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

        # Learned state — set by fit().
        self._background: Optional[np.ndarray] = None

    # ---- Internal resolvers ----------------------------------------------

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
        """Effective Gaussian σ in pixels (after physical→pixel conversion)."""
        return self._sigma

    @property
    def kernel_size(self) -> int:
        """Effective Gaussian kernel size in pixels (odd, ≥ 3)."""
        return self._kernel_size

    @property
    def background(self) -> Optional[np.ndarray]:
        """Learned mean-field background B(x, y), or None if not yet fit."""
        return None if self._background is None else self._background.copy()

    # ---- Internal building blocks ----------------------------------------

    def _smooth(self, frame_data: np.ndarray) -> np.ndarray:
        """Stage 1: 2-D Gaussian smoothing (cv2.GaussianBlur)."""
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
                f"{self.sensor_profile.name} expected (H, W) = ({eh}, {ew}).".format(  # type: ignore[union-attr]
                )
            )

    # ---- Public API ------------------------------------------------------

    def fit(
        self,
        X: Union[Iterable[Frame], np.ndarray],
        y: None = None,
    ) -> "TatenoPipeline":
        """Compute the mean-field background B(x, y) from calibration frames.

        Args:
            X: Either an iterable of `Frame`s (each (H, W) of pixel data) or
               a stacked array of shape (K, H, W). All frames must match the
               configured SensorProfile.
            y: Unused.

        Returns:
            self (chainable).
        """
        if isinstance(X, np.ndarray):
            if X.ndim != 3:
                raise ValueError(
                    f"Expected (K, H, W) ndarray; got shape {X.shape}."
                )
            stacked = X.astype(np.float32)
        else:
            arrays = []
            for f in X:
                self._validate_shape(f.data)
                arrays.append(f.data.astype(np.float32))
            if not arrays:
                raise ValueError("fit() received zero calibration frames.")
            stacked = np.stack(arrays, axis=0)

        # Validate shape on the stacked array too (covers ndarray path).
        self._validate_shape(stacked[0])

        # Smooth each calibration frame the same way inference frames will be
        # smoothed, then average. This keeps the subtraction symmetric.
        smoothed = np.stack([self._smooth(f) for f in stacked], axis=0)
        self._background = smoothed.mean(axis=0).astype(np.float32)
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> Frame:
        """Apply the full 3-stage pipeline to one frame.

        Returns:
            A new `Frame` whose `data` is the rectified residual. Timestamp
            and camera_id are preserved.
        """
        if not self._is_fitted or self._background is None:
            raise RuntimeError(
                "TatenoPipeline.predict() called before fit(). Run fit() with "
                "calibration frames first (empty room, no targets)."
            )
        self._validate_shape(X.data)

        smoothed = self._smooth(X.data)
        diff = smoothed - self._background
        residual = np.abs(diff).astype(np.float32)

        return Frame(
            data=residual,
            timestamp=X.timestamp,
            camera_id=X.camera_id,
            metadata={**X.metadata, "preprocessed_by": self.name},
        )

    # ---- Persistence -----------------------------------------------------

    def _state_dict(self) -> dict:
        return {
            "background": self._background,
            "resolved_sigma": self._sigma,
            "resolved_kernel_size": self._kernel_size,
        }

    def _load_state_dict(self, state: dict) -> None:
        bg = state.get("background")
        if bg is not None:
            self._background = np.asarray(bg, dtype=np.float32)
            self._is_fitted = True
        # Restore resolved values too (in case the user changed defaults
        # between save and load — we want the loaded behavior to match the
        # saved behavior).
        if "resolved_sigma" in state:
            self._sigma = float(state["resolved_sigma"])
        if "resolved_kernel_size" in state:
            self._kernel_size = int(state["resolved_kernel_size"])
