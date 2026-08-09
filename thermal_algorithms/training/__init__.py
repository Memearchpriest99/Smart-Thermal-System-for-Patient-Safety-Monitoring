"""Training & evaluation harness.

Reference SupervisedTrainer (PyTorch loop), dataset loaders for the labelled
data described in section 4.3, and the evaluation metrics from section 2.2.3
that drive the algorithmic performance evaluation in section 5.3.
"""

from thermal_algorithms.training.label_io import (
    ACTIVITY_LABEL_COLUMNS,
    CONTACT_LABEL_FILENAME,
    FIRE_CLASS_ID,
    PERSON_CLASS_ID,
    ActivityLabel,
    frame_index_from_label_path,
    has_contact_labels,
    list_label_files,
    load_activity_labels,
    load_classes_file,
    load_contact_labels,
    load_yolo_labels,
    save_contact_labels,
)
from thermal_algorithms.training.datasets import (
    ContactFrameDataset,
    DatasetIndex,
    FireFrameDataset,
    FrameLevelDataset,
    SessionMetadata,
    sample_background_patches,
)
from thermal_algorithms.training.hdf5_source import (
    HDF5CameraSession,
    infer_fps_from_chunk_spacing,
    list_chunks,
    read_chunk,
)
from thermal_algorithms.training.label_join import (
    LabelInterval,
    LabelJoiner,
    classify_event,
    detect_room_id,
    load_room_labels,
)
from thermal_algorithms.training.multi_source import (
    HDF5Session,
    HDF5SessionRef,
    MultiSourceContactDataset,
    MultiSourceFireDataset,
    discover_hdf5_sessions,
)
from thermal_algorithms.training.pseudo_labels import (
    PseudoLabeledFrameDataset,
    generate_pseudo_person_labels,
    load_pseudo_labels,
    merge_by_session,
    save_pseudo_labels,
)
from thermal_algorithms.training.full_corpus import (
    SYNTH_ROOMS,
    iter_contact_training_chunks,
    iter_synth_sessions,
    stream_contact_examples,
    stream_fire_examples,
    stream_fire_examples_subsampled,
)
from thermal_algorithms.training.split import (
    CONTACT_TEST_SCENES,
    build_task_split,
    session_train_test_split,
)
from thermal_algorithms.training.balance import (
    balance_examples,
    build_balanced_contact_pools,
    build_balanced_fire_pool,
    build_balanced_human_pool,
    interleave_chunks,
    size_and_cap_negative_runs,
    split_contact_pools_train_val,
)
from thermal_algorithms.training.trainer import Trainer
from thermal_algorithms.training.eval_report import (
    TimedEvalResult,
    evaluate_contact_timed,
    evaluate_fire_timed,
    evaluate_human_timed,
)
from thermal_algorithms.training.metrics import (
    BinaryConfusionMatrix,
    ScenarioResult,
    aggregate_scenarios,
    best_iou,
    binary_confusion_matrix,
    detections_to_binary_labels,
    format_scenario_table,
    iou_bbox,
    iou_mask_vs_bbox,
    mean_detection_iou,
    mean_sbr,
    signal_to_background_ratio,
)

__all__ = [
    # label_io
    "ACTIVITY_LABEL_COLUMNS",
    "CONTACT_LABEL_FILENAME",
    "FIRE_CLASS_ID",
    "PERSON_CLASS_ID",
    "ActivityLabel",
    "frame_index_from_label_path",
    "has_contact_labels",
    "list_label_files",
    "load_activity_labels",
    "load_classes_file",
    "load_contact_labels",
    "load_yolo_labels",
    "save_contact_labels",
    # datasets
    "ContactFrameDataset",
    "DatasetIndex",
    "FireFrameDataset",
    "FrameLevelDataset",
    "SessionMetadata",
    "sample_background_patches",
    # hdf5_source
    "HDF5CameraSession",
    "infer_fps_from_chunk_spacing",
    "list_chunks",
    "read_chunk",
    # label_join
    "LabelInterval",
    "LabelJoiner",
    "classify_event",
    "detect_room_id",
    "load_room_labels",
    # multi_source
    "HDF5Session",
    "HDF5SessionRef",
    "MultiSourceContactDataset",
    "MultiSourceFireDataset",
    "discover_hdf5_sessions",
    # pseudo_labels
    "PseudoLabeledFrameDataset",
    "generate_pseudo_person_labels",
    "load_pseudo_labels",
    "merge_by_session",
    "save_pseudo_labels",
    # full_corpus
    "SYNTH_ROOMS",
    "iter_contact_training_chunks",
    "iter_synth_sessions",
    "stream_contact_examples",
    "stream_fire_examples",
    "stream_fire_examples_subsampled",
    # split
    "CONTACT_TEST_SCENES",
    "build_task_split",
    "session_train_test_split",
    # balance
    "balance_examples",
    "build_balanced_fire_pool",
    "build_balanced_human_pool",
    "build_balanced_contact_pools",
    "split_contact_pools_train_val",
    "size_and_cap_negative_runs",
    "interleave_chunks",
    # trainer
    "Trainer",
    # eval_report
    "TimedEvalResult",
    "evaluate_contact_timed",
    "evaluate_fire_timed",
    "evaluate_human_timed",
    # metrics
    "BinaryConfusionMatrix",
    "ScenarioResult",
    "aggregate_scenarios",
    "best_iou",
    "binary_confusion_matrix",
    "detections_to_binary_labels",
    "format_scenario_table",
    "iou_bbox",
    "iou_mask_vs_bbox",
    "mean_detection_iou",
    "mean_sbr",
    "signal_to_background_ratio",
]
