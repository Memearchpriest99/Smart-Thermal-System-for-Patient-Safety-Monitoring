"""Contract tests for the CheckpointRegistry."""

from __future__ import annotations

import pytest

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.checkpoints import CheckpointRegistry, INVARIANT_KEY
from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984


# ---------------------------------------------------------------------------
# Dummies
# ---------------------------------------------------------------------------

class _Invariant(ThermalAlgorithm):
    name = "_inv_algo"
    is_trainable = True
    resolution_behavior = "invariant"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        return X


class _Fixed(ThermalAlgorithm):
    name = "_fixed_algo"
    is_trainable = True
    resolution_behavior = "fixed"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        return X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegistryPaths:
    def test_path_uses_default_for_invariant(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        p = reg.path_for("_inv_algo", profile_name=None)
        assert p.parent.name == "_inv_algo"
        assert p.stem == INVARIANT_KEY

    def test_path_uses_profile_name_for_fixed(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        p = reg.path_for("_fixed_algo", profile_name="MLX90640")
        assert p.parent.name == "_fixed_algo"
        assert p.stem == "MLX90640"


class TestRegistryRegister:
    def test_register_invariant(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        algo = _Invariant()
        algo.fit([1, 2])
        path = reg.register(algo)
        assert path.is_file()
        assert reg.exists("_inv_algo", profile_name=None)

    def test_register_fixed_uses_profile_name(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        algo = _Fixed(sensor_profile=MLX90640).fit([1])
        path = reg.register(algo)
        assert path.name.startswith("MLX90640")

    def test_register_refuses_unfitted_trainable(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        algo = _Invariant()
        with pytest.raises(RuntimeError, match="not been fitted"):
            reg.register(algo)


class TestRegistryLoad:
    def test_load_invariant_roundtrip(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        original = _Invariant(alpha=0.7).fit([1, 2])
        reg.register(original)
        loaded = reg.load(_Invariant)
        assert loaded.get_params() == {"alpha": 0.7}

    def test_load_fixed_requires_profile(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        with pytest.raises(ValueError, match="profile_name is required"):
            reg.load(_Fixed)

    def test_load_fixed_with_profile(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        original = _Fixed(sensor_profile=MLX90640, lr=0.01).fit([1])
        reg.register(original)
        loaded = reg.load(_Fixed, profile_name="MLX90640")
        assert loaded.get_params() == {"lr": 0.01}
        assert loaded.sensor_profile is MLX90640

    def test_load_missing_raises(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.load(_Fixed, profile_name="MLX90640")

    def test_two_profiles_coexist(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        a = _Fixed(sensor_profile=MLX90640, tag="mlx").fit([1])
        b = _Fixed(sensor_profile=WAVESHARE_26984, tag="ws").fit([1])
        reg.register(a)
        reg.register(b)
        loaded_mlx = reg.load(_Fixed, profile_name="MLX90640")
        loaded_ws = reg.load(_Fixed, profile_name="Waveshare_26984")
        assert loaded_mlx.get_params()["tag"] == "mlx"
        assert loaded_ws.get_params()["tag"] == "ws"


class TestRegistryDiscovery:
    def test_list_available_yields_registered(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path)
        reg.register(_Invariant().fit([1]))
        reg.register(_Fixed(sensor_profile=MLX90640).fit([1]))
        available = list(reg.list_available())
        assert ("_inv_algo", None) in available
        assert ("_fixed_algo", "MLX90640") in available

    def test_list_available_empty_when_root_missing(self, tmp_path):
        reg = CheckpointRegistry(root=tmp_path / "doesnotexist")
        # The constructor creates the directory; verify empty listing.
        assert list(reg.list_available()) == []
