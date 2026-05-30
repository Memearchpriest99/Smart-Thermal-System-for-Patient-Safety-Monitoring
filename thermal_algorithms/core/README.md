# core

Foundation layer shared by every other module.

| File | Purpose |
|---|---|
| `types.py` | Frozen dataclasses: `Frame`, `Detection`, `FireAlert`, `ContactEvent`, `HomographyMatrices` |
| `base.py` | `ThermalAlgorithm` ABC — `fit`, `predict`, `save`, `load`, `reset`, `get_params` |
| `sensor_profile.py` | `SensorProfile` dataclass + `MLX90640` and `WAVESHARE_26984` constants |
| `checkpoints.py` | `CheckpointRegistry` — filesystem-backed `(algo_name, profile_name) → .thalg` |
| `io.py` | CSV frame I/O matching the Data Acquisition Software format |

## Key constraint

All types in `types.py` are **frozen** (immutable). Algorithms must never mutate their inputs — return new objects instead.

## Checkpoint layout

```
checkpoints/<algorithm_name>/<profile_name>.thalg
checkpoints/<algorithm_name>/_default.thalg   # resolution_behavior="invariant"
```
