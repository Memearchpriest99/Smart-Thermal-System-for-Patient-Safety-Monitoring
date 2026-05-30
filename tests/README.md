# tests

## Running

```bash
# Full working suite (no heavy dependencies needed)
python -m pytest tests/test_types.py tests/test_sensor_profile.py tests/test_base.py \
  tests/test_checkpoints.py tests/test_tateno_pipeline.py tests/test_otsu_pipeline.py \
  tests/test_fire_svm.py tests/test_contact_multiview.py tests/test_contact_geometric.py \
  tests/test_contact_mv_stgcn.py tests/test_contact_thermo_x3d.py tests/test_metrics.py \
  tests/test_label_io_extended.py tests/test_fire_contact_datasets.py \
  tests/test_trainer.py tests/test_pipeline.py

# Single file
python -m pytest tests/test_pipeline.py -v

# Single test
python -m pytest tests/test_pipeline.py::TestRestrictedAreaPath::test_human_detected_triggers_alert -v
```

## Dependency-gated tests

Some test files require optional packages that may not be installed:

| File | Requires |
|---|---|
| `test_adaptive_threshold.py`, `test_hog_svm.py` | `scikit-image` |
| `test_mobilenet_ssd.py` | `torch` |
| `test_contact_mv_stgcn.py`, `test_contact_thermo_x3d.py` | `torch` (auto-skipped via `pytest.importorskip`) |

## Synthetic data

All tests use **synthetic data** — no real dataset required. On-disk session fixtures (for dataset loader tests) are built in `tmp_path` using helpers from `test_fire_contact_datasets.py`.
