"""Core abstractions and shared types.

Everything downstream depends on this module. Nothing here depends on anything
outside `thermal_algorithms.core` itself plus numpy + standard library.

Task-specific ABCs (Preprocessor, HumanDetector, ContactDetector, FireDetector)
live in their respective task packages — import them from there to avoid
circular dependencies.
"""

from thermal_algorithms.core.base import (
    ThermalAlgorithm,
    ResolutionBehavior,
)
from thermal_algorithms.core.types import (
    Frame,
    Detection,
    ActorPosition,
    ContactEvent,
    FireAlert,
    FireLevel,
    HomographyMatrices,
)
from thermal_algorithms.core.sensor_profile import (
    SensorProfile,
    MLX90640,
    WAVESHARE_26984,
    PROFILES,
)
from thermal_algorithms.core.checkpoints import CheckpointRegistry

__all__ = [
    # Universal base
    "ThermalAlgorithm",
    "ResolutionBehavior",
    # Data types
    "Frame",
    "Detection",
    "ActorPosition",
    "ContactEvent",
    "FireAlert",
    "FireLevel",
    "HomographyMatrices",
    # Sensor profiles
    "SensorProfile",
    "MLX90640",
    "WAVESHARE_26984",
    "PROFILES",
    # Checkpoint registry
    "CheckpointRegistry",
]
