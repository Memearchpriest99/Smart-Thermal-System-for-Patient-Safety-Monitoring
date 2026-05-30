"""ThermalAlgorithm — the universal abstract base for every algorithm in this
library.

Contract overview
-----------------
Every concrete algorithm — whether it's a classical CV pipeline, a feature-based
SVM, or a PyTorch CNN — exposes the same surface:

    algo = ConcreteAlgorithm(sensor_profile=MLX90640, **hyperparams)
    algo.fit(X_train, y_train)        # training OR calibration
    y_pred = algo.predict(X_test)     # inference
    algo.save("checkpoint.pkl")
    same = ConcreteAlgorithm.load("checkpoint.pkl")

For non-learnable algorithms (e.g., adaptive thresholding), `fit()` is the
**calibration entry point**: it learns the background model, sets noise floors,
or solves the homography from marker correspondences. Algorithms with truly
nothing to calibrate return `self` from `fit()` unchanged.

Class metadata
--------------
Each concrete class declares three class-level attributes:

    name                  : stable string used as the key in the checkpoint
                            registry. Must be unique.
    is_trainable          : True for ML/DL algorithms whose `fit()` consumes
                            labelled data. False for pure CV pipelines.
    resolution_behavior   : 'invariant' | 'parameterized' | 'fixed'
                            (see core/sensor_profile.py for the meaning).

These exist so callers (e.g., the training harness, the checkpoint registry,
section 5.3 benchmark drivers) can introspect what to do with the class
without having to instantiate it.

Persistence
-----------
The default `save() / load()` writes a single pickle file containing:

    {
        "format_version": _SAVED_FORMAT_VERSION,
        "class_name": fully-qualified class path,
        "library_version": thermal_algorithms.__version__,
        "sensor_profile_name": profile.name or None,
        "params": self.get_params(),
        "state": self._state_dict(),
    }

Concrete classes override `_state_dict()` / `_load_state_dict()` to add their
learned parameters (e.g., a PyTorch state_dict, an sklearn estimator object).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from thermal_algorithms.core.sensor_profile import SensorProfile, PROFILES


ResolutionBehavior = Literal["invariant", "parameterized", "fixed"]
"""How a concrete algorithm reacts to switching sensor profiles.

    'invariant'     — same code, same checkpoint, both sensors work.
    'parameterized' — same architecture, kernel/grid sizes scale with the profile.
    'fixed'         — one trained checkpoint per resolution.
"""


_SAVED_FORMAT_VERSION: int = 1
"""Bump when changing the save/load wire format. `load()` raises on mismatch."""


class ThermalAlgorithm(ABC):
    """Universal abstract base.

    Subclasses MUST:
        * Declare `name` as a unique class-level string.
        * Implement `fit(X, y=None)` and `predict(X)`.

    Subclasses MAY:
        * Override `_state_dict()` / `_load_state_dict()` to persist learned
          state beyond the constructor hyperparameters.
        * Override `reset()` if they hold internal state across calls (e.g.,
          temporal buffers in the Otsu fire pipeline).
        * Override `_validate_profile()` to refuse profiles they can't serve
          (e.g., a checkpoint trained only on the MLX90640 profile).
    """

    # ---- Class-level metadata (override in subclasses) -------------------

    name: ClassVar[str] = "<unnamed>"
    """Stable identifier. Used as the key in CheckpointRegistry."""

    is_trainable: ClassVar[bool] = False
    """True if `fit()` consumes labelled training data and updates learned
    parameters. False for classical CV pipelines (where `fit()` is at most
    a calibration step)."""

    resolution_behavior: ClassVar[ResolutionBehavior] = "invariant"
    """How the algorithm reacts to a SensorProfile change."""

    # ---- Construction -----------------------------------------------------

    def __init__(
        self,
        sensor_profile: Optional[SensorProfile] = None,
        **params: Any,
    ) -> None:
        """
        Args:
            sensor_profile: The sensor this instance is configured for.
                Required for 'parameterized' and 'fixed' algorithms; ignored
                by 'invariant' ones but accepted for uniform construction.
            **params: Hyperparameters specific to the concrete algorithm.
                Stored on `self` so `get_params()` / `set_params()` round-trip.
        """
        self._validate_profile(sensor_profile)
        self._sensor_profile: Optional[SensorProfile] = sensor_profile
        self._params: dict[str, Any] = dict(params)
        self._is_fitted: bool = False

    def _validate_profile(self, profile: Optional[SensorProfile]) -> None:
        """Refuse profiles the algorithm cannot serve.

        The default implementation enforces only that 'parameterized' and
        'fixed' algorithms get *some* profile. Subclasses can override to
        restrict to a specific name (e.g., a checkpoint that was only trained
        on MLX90640).
        """
        if self.resolution_behavior in {"parameterized", "fixed"} and profile is None:
            raise ValueError(
                f"{type(self).__name__} has resolution_behavior="
                f"{self.resolution_behavior!r} and requires a SensorProfile."
            )

    # ---- Properties -------------------------------------------------------

    @property
    def sensor_profile(self) -> Optional[SensorProfile]:
        return self._sensor_profile

    @property
    def is_fitted(self) -> bool:
        """True once `fit()` has been called at least once.

        Concrete classes should set `self._is_fitted = True` at the end of a
        successful `fit()`. `predict()` may use this to raise a clear error
        when called on an unfit estimator.
        """
        return self._is_fitted

    # ---- Core abstract methods -------------------------------------------

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> "ThermalAlgorithm":
        """Train or calibrate.

        For trainable algorithms, consumes labelled data and updates learned
        parameters. For non-trainable algorithms, consumes calibration data
        (e.g., background frames, homography markers) or does nothing.

        Returns:
            self, for method chaining (matches the sklearn convention).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Run inference on a single input or a batch.

        Task-specific ABCs (Preprocessor, HumanDetector, ContactDetector,
        FireDetector) narrow this signature to a typed contract.
        """
        raise NotImplementedError

    # ---- Hyperparameters (sklearn-shaped) --------------------------------

    def get_params(self) -> dict[str, Any]:
        """Return a fresh dict of the constructor hyperparameters.

        Used by save/load to persist the configuration and by the training
        harness to log experiment settings.
        """
        return dict(self._params)

    def set_params(self, **params: Any) -> "ThermalAlgorithm":
        """Update hyperparameters in place. Returns self for chaining.

        Concrete classes that need to re-derive internal state when a
        hyperparameter changes (e.g., rebuilding a Gaussian kernel) should
        override this to trigger that recomputation, then call super().
        """
        self._params.update(params)
        return self

    # ---- State (override for learned parameters) -------------------------

    def _state_dict(self) -> dict[str, Any]:
        """Return algorithm-specific learned state for persistence.

        Default: empty dict (suitable for pure classical algorithms with no
        learned state).

        Subclasses override to include:
            * sklearn estimators (pickle-serializable directly)
            * PyTorch state_dict
            * Calibrated background frames, homography matrices, etc.
        """
        return {}

    def _load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore algorithm-specific learned state.

        Default: assert that the saved state is empty. Concrete classes
        override to load their learned parameters.
        """
        if state:
            raise ValueError(
                f"{type(self).__name__} does not override _load_state_dict but "
                f"received non-empty state: keys={list(state)}"
            )

    # ---- Statefulness ----------------------------------------------------

    def reset(self) -> None:
        """Clear any in-memory state that accumulates across `predict()` calls.

        Default: no-op. Override in stateful algorithms — notably the Otsu
        fire pipeline, which carries a temporal mass-gradient buffer S_n
        across frames (§ 4.4.4 step 3).
        """
        return None

    # ---- Persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist hyperparameters + learned state to a single file.

        The format is intentionally simple — pickle of a metadata dict —
        because we want one save path for every concrete class regardless
        of whether its state is a sklearn object, a torch state_dict, or
        empty. Concrete classes customize via `_state_dict()`.
        """
        import thermal_algorithms

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": _SAVED_FORMAT_VERSION,
            "class_name": f"{type(self).__module__}.{type(self).__name__}",
            "library_version": thermal_algorithms.__version__,
            "sensor_profile_name": (
                self._sensor_profile.name if self._sensor_profile else None
            ),
            "params": self.get_params(),
            "state": self._state_dict(),
            "is_fitted": self._is_fitted,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "ThermalAlgorithm":
        """Restore an instance from a file written by `save()`.

        Verifies the saved class name matches `cls` so accidental cross-loading
        (e.g., loading a HOGSVMDetector checkpoint as MobileNetSSDDetector)
        fails loudly with a clear error.
        """
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        if payload.get("format_version") != _SAVED_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported save format version "
                f"{payload.get('format_version')!r} (expected "
                f"{_SAVED_FORMAT_VERSION!r}). File: {path}"
            )

        expected_class = f"{cls.__module__}.{cls.__name__}"
        if payload["class_name"] != expected_class:
            raise TypeError(
                f"Checkpoint was saved by {payload['class_name']!r} but "
                f"{expected_class!r} is being loaded. Use the original class "
                f"or the CheckpointRegistry to resolve the right one."
            )

        profile = PROFILES.get(payload["sensor_profile_name"]) \
            if payload["sensor_profile_name"] else None

        instance = cls(sensor_profile=profile, **payload["params"])
        instance._load_state_dict(payload["state"])
        instance._is_fitted = bool(payload.get("is_fitted", False))
        return instance

    # ---- Misc ------------------------------------------------------------

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        profile_str = (
            f"sensor={self._sensor_profile.name!r}"
            if self._sensor_profile is not None
            else "sensor=None"
        )
        return f"{type(self).__name__}({profile_str}{', ' if params_str else ''}{params_str})"
