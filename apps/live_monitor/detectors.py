"""Per-task detector registry with live-swappable options.

Each task (fire / person / touch) exposes a list of :class:`DetectorOption`s.
Rule-based detectors are always available; trainable ones are available only
when both their dependency imports succeed (torch / scikit-image) *and* a
checkpoint exists for the active sensor profile. The UI greys out unavailable
options instead of crashing.

This module is Qt-free and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.checkpoints import CheckpointRegistry
from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import HomographyMatrices

TASKS = ("fire", "person", "touch")


class DetectorUnavailable(RuntimeError):
    """Raised when an option cannot be built (missing dep or checkpoint)."""


@dataclass
class BuildContext:
    """Everything a factory might need to construct a detector."""

    profile: SensorProfile
    registry: CheckpointRegistry
    homography: Optional[HomographyMatrices] = None
    epsilon_m: float = 0.5
    delta_m: float = 0.5

    def homography_or_identity(self) -> HomographyMatrices:
        if self.homography is not None:
            return self.homography
        eye = np.eye(3, dtype=np.float64)
        return HomographyMatrices(h1=eye.copy(), h2=eye.copy(), h3=eye.copy())


@dataclass
class DetectorOption:
    """One selectable algorithm for a task."""

    task: str
    key: str                       # algorithm `name` (registry key)
    display: str                   # label shown in the combo box
    trainable: bool
    _build: Callable[[BuildContext], ThermalAlgorithm]
    _availability: Callable[[BuildContext], tuple[bool, str]]
    metadata: dict = field(default_factory=dict)

    def availability(self, ctx: BuildContext) -> tuple[bool, str]:
        """Return ``(available, reason)``. ``reason`` is empty when available."""
        try:
            return self._availability(ctx)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def build(self, ctx: BuildContext) -> ThermalAlgorithm:
        ok, reason = self.availability(ctx)
        if not ok:
            raise DetectorUnavailable(f"{self.display}: {reason}")
        return self._build(ctx)


# --- availability helpers -------------------------------------------------

def _always_available(_ctx: BuildContext) -> tuple[bool, str]:
    return True, ""


def _checkpoint_available(cls: type[ThermalAlgorithm]) -> Callable[[BuildContext], tuple[bool, str]]:
    def check(ctx: BuildContext) -> tuple[bool, str]:
        profile_name = None if cls.resolution_behavior == "invariant" else ctx.profile.name
        if not ctx.registry.exists(cls.name, profile_name):
            tag = profile_name or "default"
            return False, f"no checkpoint ({cls.name}/{tag})"
        return True, ""
    return check


def _import_and_checkpoint(
    importer: Callable[[], type[ThermalAlgorithm]],
    dep_label: str,
) -> Callable[[BuildContext], tuple[bool, str]]:
    def check(ctx: BuildContext) -> tuple[bool, str]:
        try:
            cls = importer()
        except Exception:
            return False, f"{dep_label} not installed"
        return _checkpoint_available(cls)(ctx)
    return check


# --- lazy importers (keep torch / skimage off the import path) ------------

def _imp_fire_svm() -> type[ThermalAlgorithm]:
    from thermal_algorithms.fire_detection.fire_svm import FireSVMDetector
    return FireSVMDetector


def _imp_hog_svm() -> type[ThermalAlgorithm]:
    from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
    return HOGSVMDetector


def _imp_mobilenet() -> type[ThermalAlgorithm]:
    from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
    return MobileNetSSDDetector


def _imp_mvstgcn() -> type[ThermalAlgorithm]:
    from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector
    return MVSTGCNDetector


def _imp_thermox3d() -> type[ThermalAlgorithm]:
    from thermal_algorithms.contact_detection.thermo_x3d import ThermoX3DDetector
    return ThermoX3DDetector


# --- builders -------------------------------------------------------------

def _fitted(algo: ThermalAlgorithm) -> ThermalAlgorithm:
    """Mark rule-based detectors fitted (their ``fit`` is a no-op calibration)."""
    if not algo.is_fitted:
        algo.fit([])
    return algo


def _build_otsu(ctx: BuildContext) -> ThermalAlgorithm:
    from thermal_algorithms.fire_detection.otsu_pipeline import OtsuFireDetector
    return _fitted(OtsuFireDetector(ctx.profile))


def _build_adaptive(ctx: BuildContext) -> ThermalAlgorithm:
    from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
    return _fitted(AdaptiveThresholdDetector(ctx.profile))


def _build_geometric(ctx: BuildContext) -> ThermalAlgorithm:
    from thermal_algorithms.contact_detection.geometric import GeometricContactDetector
    det = GeometricContactDetector(
        ctx.profile,
        homography=ctx.homography_or_identity(),
        epsilon_m=ctx.epsilon_m,
        delta_m=ctx.delta_m,
    )
    return _fitted(det)


def _load_checkpoint(importer: Callable[[], type[ThermalAlgorithm]]) -> Callable[[BuildContext], ThermalAlgorithm]:
    def build(ctx: BuildContext) -> ThermalAlgorithm:
        cls = importer()
        profile_name = None if cls.resolution_behavior == "invariant" else ctx.profile.name
        algo = ctx.registry.load(cls, profile_name=profile_name)
        # Geometric-free contact nets still need a homography for fusion.
        if hasattr(algo, "set_homography") and ctx.homography is not None:
            try:
                algo.set_homography(ctx.homography)
            except Exception:
                pass
        return algo
    return build


# --- the registry ---------------------------------------------------------

def build_registry() -> dict[str, list[DetectorOption]]:
    """Return ``{task: [options...]}`` with the rule-based option first."""
    return {
        "fire": [
            DetectorOption("fire", "otsu_fire_detector", "Otsu pipeline (rule-based)",
                           False, _build_otsu, _always_available),
            DetectorOption("fire", "fire_svm_detector", "Fire SVM (trained)",
                           True, _load_checkpoint(_imp_fire_svm),
                           _import_and_checkpoint(_imp_fire_svm, "scikit-learn")),
        ],
        "person": [
            DetectorOption("person", "adaptive_threshold_detector", "Adaptive threshold (rule-based)",
                           False, _build_adaptive, _always_available),
            DetectorOption("person", "hog_svm_detector", "HOG + SVM (trained)",
                           True, _load_checkpoint(_imp_hog_svm),
                           _import_and_checkpoint(_imp_hog_svm, "scikit-image")),
            DetectorOption("person", "mobilenet_ssd_detector", "MobileNet-SSD (deep)",
                           True, _load_checkpoint(_imp_mobilenet),
                           _import_and_checkpoint(_imp_mobilenet, "torch")),
        ],
        "touch": [
            DetectorOption("touch", "geometric_contact_detector", "Geometric fusion (rule-based)",
                           False, _build_geometric, _always_available,
                           metadata={"birdseye": True}),
            DetectorOption("touch", "mv_stgcn_detector", "MV-STGCN (deep)",
                           True, _load_checkpoint(_imp_mvstgcn),
                           _import_and_checkpoint(_imp_mvstgcn, "torch")),
            DetectorOption("touch", "thermo_x3d_detector", "Thermo-X3D (deep)",
                           True, _load_checkpoint(_imp_thermox3d),
                           _import_and_checkpoint(_imp_thermox3d, "torch")),
        ],
    }


def default_checkpoint_root() -> Path:
    """Repo-root ``checkpoints/`` directory (created if missing by the registry)."""
    # apps/live_monitor/detectors.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[2] / "checkpoints"


# --- live threshold tuning ------------------------------------------------

# Maps a friendly knob name to the private attribute(s) a detector may expose.
_THRESHOLD_ATTRS = {
    "t_ign": ("_t_ign",),
    "t_fire": ("_t_fire",),
    "score_threshold": ("_score_threshold",),
    "delta_m": ("_delta_m",),
    "epsilon_m": ("_epsilon_m",),
}


def apply_thresholds(detector: ThermalAlgorithm, **knobs: float) -> dict[str, float]:
    """Best-effort live tuning: set known private threshold attributes if the
    detector has them. Returns the knobs that were actually applied."""
    applied: dict[str, float] = {}
    for knob, value in knobs.items():
        if value is None:
            continue
        for attr in _THRESHOLD_ATTRS.get(knob, ()):
            if hasattr(detector, attr):
                setattr(detector, attr, value)
                applied[knob] = value
                break
    return applied
