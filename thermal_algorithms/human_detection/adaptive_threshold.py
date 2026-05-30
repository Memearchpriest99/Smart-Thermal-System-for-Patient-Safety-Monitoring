"""AdaptiveThresholdDetector - classical CV human detection (Section 4.4.2.1).

Pipeline:
    a. Adaptive thresholding - compute the local Gaussian-weighted mean for
       each pixel, segment regions exceeding (mean + C).
    b. Morphological closing - dilation followed by erosion to fill thermal
       gaps within a heat blob (e.g., gaps caused by clothing insulation).
    c. Geometric filtering - validate detected blobs by area, solidity, and
       aspect ratio. Optionally also gate by intensity range.

Adaptive thresholding equation (report 4.4.2.1, sign-corrected for the
intuitive "noise margin" semantics):

    D(x,y) = 1  if I(x,y) > local_mean(x,y) + C
           = 0  otherwise

Where local_mean is computed via a Gaussian kernel of size (block_size x
block_size) and C is a noise margin in input units. Larger C is stricter.

NOTE: the report writes the formula as `I > mean - C`, but its accompanying
text ("C is a constant offset to filter noise") is consistent with the sign
used here.

The local mean is computed via `cv2.GaussianBlur` (a faithful 2-D Gaussian
convolution) rather than `cv2.adaptiveThreshold` so the algorithm operates
on float32 thermal data without re-quantization to uint8.

Input data
----------
This detector is preprocessor-agnostic: it works on raw frames OR on the
Tateno-pipeline residual. The semantics of the intensity-gating parameter
`pixel_value_bounds` change accordingly:
    * Raw frames     -> bounds in absolute deg C (e.g., (28, 42) for humans)
    * Residual frames -> bounds in residual deg C above background

When `pixel_value_bounds` is None (default), no intensity gating is applied.

Resolution behavior
-------------------
'parameterized' - block_size defaults to a *physical* scale (60 cm at 2 m
range), converted to pixels via the SensorProfile. The block must be larger
than the target for adaptive thresholding to work.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.human_detection.base import HumanDetector


class AdaptiveThresholdDetector(HumanDetector):
    """Reference implementation of the 4.4.2.1 classical-CV human detector."""

    name = "adaptive_threshold_detector"
    is_trainable = False
    resolution_behavior = "parameterized"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        block_size: Optional[int] = None,
        physical_block_size_m: float = 0.60,
        assumed_distance_m: float = 2.0,
        c_offset: float = 0.5,
        morph_kernel_size: int = 3,
        min_area_pixels: int = 4,
        max_area_fraction: float = 0.5,
        min_solidity: float = 0.50,
        min_aspect_ratio: float = 0.20,
        max_aspect_ratio: float = 5.00,
        pixel_value_bounds: Optional[tuple[float, float]] = None,
    ) -> None:
        if sensor_profile is None:
            raise ValueError("AdaptiveThresholdDetector requires a SensorProfile.")
        if not 0.0 < max_area_fraction <= 1.0:
            raise ValueError(
                f"max_area_fraction must be in (0, 1]; got {max_area_fraction}."
            )

        super().__init__(
            sensor_profile=sensor_profile,
            block_size=block_size,
            physical_block_size_m=physical_block_size_m,
            assumed_distance_m=assumed_distance_m,
            c_offset=c_offset,
            morph_kernel_size=morph_kernel_size,
            min_area_pixels=min_area_pixels,
            max_area_fraction=max_area_fraction,
            min_solidity=min_solidity,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            pixel_value_bounds=pixel_value_bounds,
        )

        self._block_size = self._resolve_block_size(
            sensor_profile, block_size, physical_block_size_m, assumed_distance_m
        )
        if morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
            raise ValueError(
                f"morph_kernel_size must be a positive odd integer; "
                f"got {morph_kernel_size}."
            )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self._c_offset = float(c_offset)
        self._min_area = int(min_area_pixels)
        ew, eh = sensor_profile.resolution
        self._max_area = int(max_area_fraction * ew * eh)
        self._min_solidity = float(min_solidity)
        self._min_aspect = float(min_aspect_ratio)
        self._max_aspect = float(max_aspect_ratio)
        self._pixel_bounds = (
            None if pixel_value_bounds is None
            else (float(pixel_value_bounds[0]), float(pixel_value_bounds[1]))
        )

    @staticmethod
    def _resolve_block_size(
        profile: SensorProfile,
        block_size: Optional[int],
        physical_block_size_m: float,
        assumed_distance_m: float,
    ) -> int:
        if block_size is None:
            if physical_block_size_m <= 0 or assumed_distance_m <= 0:
                raise ValueError(
                    "physical_block_size_m and assumed_distance_m must be positive."
                )
            dx, dy = profile.physical_pixel_size_m(assumed_distance_m)
            avg = (dx + dy) / 2.0
            block_size = max(3, int(round(physical_block_size_m / avg)))
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        return int(block_size)

    @property
    def block_size(self) -> int:
        return self._block_size

    def _validate_shape(self, data: np.ndarray) -> None:
        ew, eh = self.sensor_profile.resolution
        if data.shape != (eh, ew):
            raise ValueError(
                f"Frame shape {data.shape} does not match "
                f"{self.sensor_profile.name} expected (H, W) = ({eh}, {ew})."
            )

    def _segment(self, image: np.ndarray) -> np.ndarray:
        local_mean = cv2.GaussianBlur(
            image.astype(np.float32),
            ksize=(self._block_size, self._block_size),
            sigmaX=0,
            borderType=cv2.BORDER_REFLECT,
        )
        mask = (image > local_mean + self._c_offset).astype(np.uint8) * 255
        return mask

    def _close(self, mask: np.ndarray) -> np.ndarray:
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)

    def _solidity(self, contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        return float(area / hull_area) if hull_area > 0 else 0.0

    def _extract_thermal_features(
        self, image: np.ndarray, mask: np.ndarray
    ) -> dict[str, float]:
        pixels = image[mask > 0]
        return {
            "max_temp": float(pixels.max()),
            "mean_temp": float(pixels.mean()),
            "std_temp": float(pixels.std()),
            "area_pixels": float(pixels.size),
        }

    def fit(
        self,
        X: Iterable[Frame],
        y: Optional[Iterable[list[Detection]]] = None,
    ) -> "AdaptiveThresholdDetector":
        self._is_fitted = True
        return self

    def predict(self, X: Frame) -> list[Detection]:
        self._validate_shape(X.data)

        raw_mask = self._segment(X.data)
        closed_mask = self._close(raw_mask)

        contours, _ = cv2.findContours(
            closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self._min_area or area > self._max_area:
                continue

            solidity = self._solidity(contour)
            if solidity < self._min_solidity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect = w / h
            if aspect < self._min_aspect or aspect > self._max_aspect:
                continue

            blob_mask = np.zeros_like(closed_mask)
            cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
            features = self._extract_thermal_features(X.data, blob_mask)
            features["solidity"] = solidity
            features["aspect_ratio"] = float(aspect)

            if self._pixel_bounds is not None:
                lo, hi = self._pixel_bounds
                if not (lo <= features["mean_temp"] <= hi):
                    continue

            detections.append(
                Detection(
                    bbox=(float(x), float(y), float(w), float(h)),
                    score=1.0,
                    class_id=0,
                    camera_id=X.camera_id,
                    thermal_features=features,
                )
            )

        return detections

    def _state_dict(self) -> dict:
        return {"resolved_block_size": self._block_size}

    def _load_state_dict(self, state: dict) -> None:
        if "resolved_block_size" in state:
            self._block_size = int(state["resolved_block_size"])
