"""Shared data types — frozen, lightweight, serializable.

These dataclasses are the lingua franca between algorithms. Any concrete class
implementing one of the task ABCs accepts and returns these types only, which
is what makes alternative algorithms interchangeable at the call site.

Design rules:
    * All dataclasses are frozen (immutable). Algorithms must not mutate their
      inputs.
    * No PyTorch tensors at this layer — only numpy + python primitives.
      Concrete trainable algorithms convert internally.
    * `to_dict()` / `from_dict()` round-trip cleanly for save/load.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    """A single thermal frame from one sensor.

    Attributes:
        data: 2-D thermal array of shape (H, W). Values are temperature in °C
            (or normalized; consult the producing pipeline). dtype is typically
            float32 after preprocessing, may be uint16 / float for raw input.
        timestamp: Capture time in seconds (epoch or monotonic — consult the
            producing pipeline). Used by the Kalman tracker to correct for the
            TCA9548A I2C phase shift between cameras.
        camera_id: Which physical sensor produced this frame, in {0, 1, 2}.
            Optional for single-sensor pipelines.
        metadata: Free-form per-frame info (e.g., sensor temperature, gain).
            Not used by the core algorithms; provided for tracing.
    """

    data: np.ndarray
    timestamp: float
    camera_id: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(
                f"Frame.data must be 2-D (H, W); got shape {self.data.shape}."
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Detection (output of HumanDetector)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single detected object in a frame.

    The bbox convention is (x, y, w, h) with (x, y) at the top-left corner,
    matching the foot-point formula in § 4.4.3 Stage 1:
        P_foot = (x + w/2, y + h)

    Attributes:
        bbox: (x, y, w, h) in pixel coordinates of the source frame.
        score: Detection confidence in [0, 1]. Classical detectors that do not
            produce a probability output use 1.0.
        class_id: 0 = person (only class for human detection in this project).
            Reserved for future multi-class extension.
        camera_id: Which camera this detection comes from. Required for
            multi-view fusion; optional for single-view evaluation.
        velocity: Optional (vx, vy) in pixels/second from the Kalman tracker.
            Populated by the MV-STGCN pipeline after tracking; None for raw
            single-frame detectors.
        thermal_features: Optional precomputed features over the bbox region
            (max_temp, mean_temp, std, area). Hoisted out so FireDetector and
            ContactDetector can reuse them without re-walking the pixels.
    """

    bbox: tuple[float, float, float, float]
    score: float = 1.0
    class_id: int = 0
    camera_id: Optional[int] = None
    velocity: Optional[tuple[float, float]] = None
    thermal_features: Optional[dict[str, float]] = None

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-center of the bbox; per § 4.4.3 this approximates the
        subject's contact point with the floor."""
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h)

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> float:
        _, _, w, h = self.bbox
        return float(w * h)


# ---------------------------------------------------------------------------
# Contact detection output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActorPosition:
    """A unique subject's location on the ground plane after multi-view fusion.

    Attributes:
        world_xy: (X, Y) in meters on the room's floor plane (Z=0). Result of
            projecting `Detection.foot_point` through the per-camera homography
            and clustering across views.
        track_id: Stable identifier across frames (assigned by the tracker).
        source_camera_ids: Which cameras contributed to this estimate. Used to
            distinguish high-confidence multi-view triangulations from
            single-view fallbacks.
        confidence: Aggregate confidence after fusion in [0, 1].
    """

    world_xy: tuple[float, float]
    track_id: int
    source_camera_ids: tuple[int, ...] = ()
    confidence: float = 1.0


@dataclass(frozen=True)
class ContactEvent:
    """Output of a ContactDetector for a single time step.

    Attributes:
        actors: All unique subjects identified in the scene.
        pairs_in_contact: Indices (i, j) into `actors` for any pair flagged as
            being in contact. Pairs are ordered (i < j) and unique.
        timestamp: Time of the inference (typically aligned to camera 3, the
            last-read sensor in the I2C chain).
        confidence: Overall confidence of the contact assertion in [0, 1].
            For deterministic pipelines (Geometric), this is 1.0 when any pair
            is in contact and 0.0 otherwise. For GCN/X3D, this is the softmax
            score.
        debug: Algorithm-specific debug payload (e.g., pairwise distances).
            Not part of the formal contract; consumers may ignore it.
    """

    actors: tuple[ActorPosition, ...]
    pairs_in_contact: tuple[tuple[int, int], ...]
    timestamp: float
    confidence: float = 1.0
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def any_contact(self) -> bool:
        return len(self.pairs_in_contact) > 0


# ---------------------------------------------------------------------------
# Fire detection output
# ---------------------------------------------------------------------------

class FireLevel(Enum):
    """Classification states from § 4.4.4.

    The pipelined approach produces all four levels. The SVM alternative
    collapses to {SAFE, ACTIVE_COMBUSTION}.
    """

    SAFE = "safe"
    POTENTIAL_FIRE = "potential_fire"        # large hot blob, awaiting growth verification
    IGNITION_SOURCE = "ignition_source"      # small concentrated heat — immediate alert
    ACTIVE_COMBUSTION = "active_combustion"  # growth confirmed by temporal mass gradient


@dataclass(frozen=True)
class FireAlert:
    """Output of a FireDetector for a single time step.

    Attributes:
        level: Classification result.
        timestamp: Time of the inference.
        blob_features: Per-blob attributes used by the decision (area, max
            temp, growth gradient G_n, persistence counter S_n, etc.). Same
            dict shape regardless of which alternative produced it.
        confidence: Probability output for ML alternatives (SVM decision
            margin → sigmoid); 1.0 for the deterministic pipeline.
        debug: Algorithm-specific debug payload.
    """

    level: FireLevel
    timestamp: float
    blob_features: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def is_alarm(self) -> bool:
        return self.level in {FireLevel.IGNITION_SOURCE, FireLevel.ACTIVE_COMBUSTION}


# ---------------------------------------------------------------------------
# Homography (input to ContactDetector)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomographyMatrices:
    """Per-camera image-plane → ground-plane projection matrices.

    Each H_k is a (3, 3) matrix that maps homogeneous pixel coordinates
    (u, v, 1) to homogeneous world coordinates (wX_w, wY_w, w) on the room's
    floor (Z=0). See § 4.4.3 "Mathematical Formulation: Homography and
    Projection".

    Calibration: by the "Hot Point Calibration" procedure of § 4.4.3 — at least
    4 heated targets at known floor positions, then SVD on the resulting linear
    system Ah = 0.
    """

    h1: np.ndarray
    h2: np.ndarray
    h3: np.ndarray

    def __post_init__(self) -> None:
        for name, H in (("h1", self.h1), ("h2", self.h2), ("h3", self.h3)):
            if H.shape != (3, 3):
                raise ValueError(
                    f"HomographyMatrices.{name} must be 3x3; got shape {H.shape}."
                )

    def __getitem__(self, camera_id: int) -> np.ndarray:
        """Lookup by camera_id ∈ {0, 1, 2}."""
        return (self.h1, self.h2, self.h3)[camera_id]

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.h1, self.h2, self.h3)
