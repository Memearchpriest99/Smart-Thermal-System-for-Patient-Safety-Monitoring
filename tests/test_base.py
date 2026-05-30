"""Contract tests for the universal ThermalAlgorithm ABC.

Uses dummy concrete subclasses to exercise the interface end-to-end without
depending on any of the actual algorithm implementations.
"""

from __future__ import annotations

import pytest

from thermal_algorithms.core.base import ThermalAlgorithm
from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984


# ---------------------------------------------------------------------------
# Dummy concretes
# ---------------------------------------------------------------------------

class _InvariantNoop(ThermalAlgorithm):
    """Bare-minimum concrete: invariant, non-trainable, no learned state."""

    name = "_invariant_noop"
    is_trainable = False
    resolution_behavior = "invariant"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        return X


class _ParameterizedTrainable(ThermalAlgorithm):
    """Carries learned state — used to test state persistence."""

    name = "_param_trainable"
    is_trainable = True
    resolution_behavior = "parameterized"

    def fit(self, X, y=None):
        self._learned_weight = float(sum(X))  # type: ignore[arg-type]
        self._is_fitted = True
        return self

    def predict(self, X):
        return self._learned_weight * X  # type: ignore[operator]

    def _state_dict(self):
        return {"weight": getattr(self, "_learned_weight", None)}

    def _load_state_dict(self, state):
        if state.get("weight") is not None:
            self._learned_weight = state["weight"]


class _FixedTrainable(ThermalAlgorithm):
    """Resolution-dependent: requires a sensor profile to instantiate."""

    name = "_fixed_trainable"
    is_trainable = True
    resolution_behavior = "fixed"

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def predict(self, X):
        return X


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------

class TestABCEnforcement:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ThermalAlgorithm()  # type: ignore[abstract]

    def test_subclass_missing_fit_predict_is_abstract(self):
        class _Bad(ThermalAlgorithm):
            name = "_bad"

        with pytest.raises(TypeError):
            _Bad()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Construction / profile validation
# ---------------------------------------------------------------------------

class TestProfileValidation:
    def test_invariant_accepts_none(self):
        algo = _InvariantNoop()
        assert algo.sensor_profile is None

    def test_invariant_accepts_any_profile(self):
        algo = _InvariantNoop(sensor_profile=MLX90640)
        assert algo.sensor_profile is MLX90640

    def test_parameterized_requires_profile(self):
        with pytest.raises(ValueError, match="SensorProfile"):
            _ParameterizedTrainable()

    def test_fixed_requires_profile(self):
        with pytest.raises(ValueError, match="SensorProfile"):
            _FixedTrainable()

    def test_fixed_accepts_profile(self):
        algo = _FixedTrainable(sensor_profile=WAVESHARE_26984)
        assert algo.sensor_profile is WAVESHARE_26984


# ---------------------------------------------------------------------------
# Hyperparameter symmetry
# ---------------------------------------------------------------------------

class TestHyperparams:
    def test_get_params_returns_construction_args(self):
        algo = _InvariantNoop(alpha=0.5, beta="x")
        assert algo.get_params() == {"alpha": 0.5, "beta": "x"}

    def test_set_params_updates(self):
        algo = _InvariantNoop(alpha=0.5)
        algo.set_params(alpha=0.9, gamma=1)
        assert algo.get_params() == {"alpha": 0.9, "gamma": 1}

    def test_get_params_returns_fresh_copy(self):
        algo = _InvariantNoop(alpha=0.5)
        p = algo.get_params()
        p["alpha"] = 999
        assert algo.get_params()["alpha"] == 0.5


# ---------------------------------------------------------------------------
# fit/predict + is_fitted
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_is_fitted_false_initially(self):
        algo = _InvariantNoop()
        assert algo.is_fitted is False

    def test_is_fitted_true_after_fit(self):
        algo = _InvariantNoop()
        algo.fit([1, 2, 3])
        assert algo.is_fitted is True

    def test_fit_returns_self(self):
        algo = _InvariantNoop()
        assert algo.fit([1, 2, 3]) is algo


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_roundtrip_invariant_noop(self, tmp_path):
        algo = _InvariantNoop(alpha=0.5, beta="hi")
        algo.fit([1, 2])
        path = tmp_path / "ckpt.thalg"
        algo.save(path)
        restored = _InvariantNoop.load(path)
        assert restored.get_params() == algo.get_params()
        assert restored.is_fitted is True

    def test_roundtrip_with_learned_state(self, tmp_path):
        algo = _ParameterizedTrainable(sensor_profile=MLX90640, lr=0.01)
        algo.fit([10, 20, 30])
        assert algo.predict(2.0) == 120.0

        path = tmp_path / "ckpt.thalg"
        algo.save(path)
        restored = _ParameterizedTrainable.load(path)
        assert restored.predict(2.0) == 120.0
        assert restored.sensor_profile is MLX90640

    def test_load_rejects_class_mismatch(self, tmp_path):
        algo = _InvariantNoop()
        algo.fit([1])
        path = tmp_path / "ckpt.thalg"
        algo.save(path)
        with pytest.raises(TypeError, match="saved by"):
            _ParameterizedTrainable.load(path)

    def test_load_preserves_sensor_profile(self, tmp_path):
        algo = _FixedTrainable(sensor_profile=WAVESHARE_26984)
        algo.fit([1])
        path = tmp_path / "ckpt.thalg"
        algo.save(path)
        restored = _FixedTrainable.load(path)
        assert restored.sensor_profile is WAVESHARE_26984

    def test_load_rejects_unknown_format_version(self, tmp_path):
        import pickle
        path = tmp_path / "ckpt.thalg"
        with path.open("wb") as f:
            pickle.dump(
                {
                    "format_version": 999,
                    "class_name": "anything",
                    "library_version": "0.0.0",
                    "sensor_profile_name": None,
                    "params": {},
                    "state": {},
                },
                f,
            )
        with pytest.raises(ValueError, match="format version"):
            _InvariantNoop.load(path)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_default_is_noop(self):
        algo = _InvariantNoop()
        # Should not raise.
        algo.reset()
