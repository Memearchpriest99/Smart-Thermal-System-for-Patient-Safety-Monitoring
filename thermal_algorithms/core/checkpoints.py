"""Checkpoint registry — resolves (algorithm_name, sensor_profile_name) to
on-disk checkpoint paths.

Motivation
----------
Algorithms with `resolution_behavior='fixed'` (MobileNet-SSD, YOLO-Nano,
Thermo-X3D) hold one trained checkpoint per sensor profile. Rather than
hard-coding paths, every concrete class is associated with a `name`
(class-level constant) and the registry maps `(name, profile_name)` to
a file under a root directory.

For `resolution_behavior='invariant'` algorithms (Geometric Contact, Fire
SVM, GCN body), only the algorithm name is used (`profile_name=None`).

Layout
------
The registry lays checkpoints out on disk as:

    <root>/<algorithm_name>/<profile_name>.thalg
    <root>/<algorithm_name>/_default.thalg         # invariant case

This keeps everything human-browsable and easy to back up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from thermal_algorithms.core.base import ThermalAlgorithm


CHECKPOINT_EXTENSION = ".thalg"
INVARIANT_KEY = "_default"


class CheckpointRegistry:
    """Filesystem-backed registry of trained checkpoints.

    Usage
    -----
        registry = CheckpointRegistry(root="./checkpoints")

        # Save:
        registry.register(my_trained_detector)

        # Load (class is known, profile is known):
        loaded = registry.load(MobileNetSSDDetector, profile_name="MLX90640")

        # List what's available:
        for algo, prof in registry.list_available():
            print(f"{algo} / {prof or 'invariant'}")
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ---- Path resolution ------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def path_for(
        self,
        algorithm_name: str,
        profile_name: Optional[str] = None,
    ) -> Path:
        """Compute the canonical checkpoint path for a given (algo, profile).

        Does not check existence. Use `exists()` to verify.
        """
        key = profile_name if profile_name else INVARIANT_KEY
        return self._root / algorithm_name / f"{key}{CHECKPOINT_EXTENSION}"

    def exists(
        self,
        algorithm_name: str,
        profile_name: Optional[str] = None,
    ) -> bool:
        return self.path_for(algorithm_name, profile_name).is_file()

    # ---- Save / load ---------------------------------------------------

    def register(self, algo: ThermalAlgorithm) -> Path:
        """Save `algo` to its canonical path. Returns the path.

        Refuses to register an algorithm that has not been fitted yet — there
        would be nothing meaningful to persist.
        """
        if not algo.is_fitted and algo.is_trainable:
            raise RuntimeError(
                f"Refusing to register {type(algo).__name__}: it is trainable "
                f"but has not been fitted. Call `algo.fit(...)` first."
            )

        # Only 'invariant' algorithms share a single checkpoint across profiles;
        # 'fixed' and 'parameterized' both need per-profile keys (HOG feature
        # dim, calibrated background, etc. depend on the profile).
        profile_name = (
            algo.sensor_profile.name
            if (algo.sensor_profile is not None and algo.resolution_behavior != "invariant")
            else None
        )
        path = self.path_for(algo.name, profile_name)
        algo.save(path)
        return path

    def load(
        self,
        algorithm_cls: type[ThermalAlgorithm],
        profile_name: Optional[str] = None,
    ) -> ThermalAlgorithm:
        """Load a checkpoint into a fresh instance of `algorithm_cls`.

        For 'fixed' algorithms, `profile_name` is required. For 'invariant'
        algorithms, it's ignored. For 'parameterized' algorithms, the
        learned state may include calibration data tied to the profile, so
        `profile_name` is recommended.
        """
        if algorithm_cls.resolution_behavior == "fixed" and profile_name is None:
            raise ValueError(
                f"{algorithm_cls.__name__} has resolution_behavior='fixed'; "
                f"profile_name is required."
            )

        # Invariant algorithms always use the _default key, regardless of what
        # the caller passes.
        if algorithm_cls.resolution_behavior == "invariant":
            profile_name = None

        path = self.path_for(algorithm_cls.name, profile_name)
        if not path.is_file():
            raise FileNotFoundError(
                f"No checkpoint found at {path}. Have you trained and registered "
                f"{algorithm_cls.__name__} for profile={profile_name!r}?"
            )
        return algorithm_cls.load(path)

    # ---- Discovery -----------------------------------------------------

    def list_available(self) -> Iterator[tuple[str, Optional[str]]]:
        """Yield (algorithm_name, profile_name) for every checkpoint on disk.

        `profile_name` is None for invariant checkpoints (those keyed as
        `_default`).
        """
        if not self._root.is_dir():
            return
        for algo_dir in sorted(self._root.iterdir()):
            if not algo_dir.is_dir():
                continue
            for ckpt in sorted(algo_dir.glob(f"*{CHECKPOINT_EXTENSION}")):
                key = ckpt.stem
                profile = None if key == INVARIANT_KEY else key
                yield algo_dir.name, profile
