# training

Dataset loaders, evaluation metrics, and the trainer harness for reproducing the §5.3 report tables.

---

## Dataset classes

### FrameLevelDataset
Yields `(Frame, list[Detection])` pairs from YOLO-labeled sessions.

```python
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset, PERSON_CLASS_ID, FIRE_CLASS_ID

index = DatasetIndex("data/", sensor_profile=MLX90640)

human_ds = FrameLevelDataset(index, class_filter=[PERSON_CLASS_ID], include_negative_frames=True)
fire_bg   = FrameLevelDataset(index, scenes=["empty_room"], include_negative_frames=True, class_filter=[])
```

### FireFrameDataset
Wraps `FrameLevelDataset` with `class_filter=[FIRE_CLASS_ID]` and converts detection lists to `FireAlert` objects for `FireSVMDetector.fit()`.

### ContactFrameDataset
Reads per-session `contact_labels.csv` files and yields `(ThreeViewFrames, ContactEvent)` pairs.

**Contact label format** — one file per session, lives next to the `.npz` files:
```
frame_idx,contact
0,0
1,0
5,1
6,1
```
Only annotated frames appear; frames absent from the file are excluded from training. Annotate every frame you intend to use — no implicit defaults.

---

## Metrics

```python
from thermal_algorithms.training import (
    BinaryConfusionMatrix, binary_confusion_matrix,
    iou_bbox, iou_mask_vs_bbox, mean_detection_iou,
    signal_to_background_ratio, mean_sbr,
    ScenarioResult, format_scenario_table, aggregate_scenarios,
)
```

`BinaryConfusionMatrix` exposes `.accuracy`, `.precision`, `.recall`, `.f1`, `.false_alarm_rate`, `.correct_over_total`. Two matrices add together with `+`.

`ScenarioResult` maps directly to the rows in Tables 3 and 4 of the Engineering Report.

---

## Trainer

```python
from thermal_algorithms.training import Trainer

# §5.3.1 — Signal-to-Background Ratio
raw_sbr, proc_sbr = Trainer.evaluate_preprocessing(dataset, preprocessor)

# §5.3.2 — Human detection (Table 3)
results = Trainer.evaluate_human_detection(detector, human_ds, mode="proc", preprocessor=pp)

# §5.3.4 — Fire detection (Table 4, with IoU column)
results = Trainer.evaluate_fire_detection(detector, fire_ds, mode="proc")

print(format_scenario_table(results, include_iou=True))
```

`Trainer` calls `detector.reset()` at every session boundary automatically — no manual state management needed. The `mode` string ("raw" / "proc") is passed through to `ScenarioResult` for the Raw/Proc row pairs in the report tables.

---

## Session layout

Both folder layouts are supported by `DatasetIndex`:

```
dataset/scene_name/timestamp/ch{N}_raw_data.npz   (layout A — timestamped sessions)
dataset/scene_name/ch{N}_raw_data.npz              (layout B — direct)
```

YOLO labels live alongside the NPZ files:
```
session_dir/ch{N}_frames/frame_XXXXX.txt
session_dir/classes.txt
session_dir/contact_labels.csv   ← contact labels (if applicable)
```
