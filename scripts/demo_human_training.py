"""Small end-to-end human-detection train + inference loop on real data.

Detectors compared: AdaptiveThreshold (no training), HOG+SVM (classical ML),
MobileNet-SSD (deep). Scenes: the four reviewed setup1 recordings
(2pplfight, 2pplwithhairdryer, 2pplwithtouch, 3ppl) PLUS empty-room background
frames as true negatives.

Pipeline:
  1. Split the empty-room scene per channel: even frames -> Tateno calibration,
     odd frames -> negative examples (no leak between calibration and eval).
  2. Build (Frame, person-boxes) for the 4 scenes + (Frame, []) for empty negs,
     each preprocessed with its channel's Tateno background.
  3. Split by frame index (every 5th -> test) so a frame's 3 views stay together.
  4. Train HOG+SVM and MobileNet-SSD on the train split.
  5. Evaluate all three on the test split: box-level (IoU>=0.3) + frame-level
     presence (now meaningful thanks to the empty negatives).

Read-only w.r.t. the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.human_detection.adaptive_threshold import AdaptiveThresholdDetector
from thermal_algorithms.human_detection.hog_svm import HOGSVMDetector
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.preprocessing.tateno_pipeline import TatenoPipeline
from thermal_algorithms.training import DatasetIndex, FrameLevelDataset, PERSON_CLASS_ID
from thermal_algorithms.training.metrics import binary_confusion_matrix, iou_bbox

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SCENES = ["2pplfight", "2pplwithhairdryer", "2pplwithtouch", "3ppl"]
EMPTY = "emptyroomwithpcscreen"
CHANNELS = (0, 1, 2)
TEST_EVERY = 5
IOU_MATCH = 0.3


def box_level_counts(preds, gts, thr=IOU_MATCH):
    matched = set(); tp = 0
    for p in preds:
        best_j, best_iou = -1, thr
        for j, g in enumerate(gts):
            if j in matched:
                continue
            i = iou_bbox(p.bbox, g.bbox)
            if i >= best_iou:
                best_iou, best_j = i, j
        if best_j >= 0:
            matched.add(best_j); tp += 1
    return tp, len(preds) - tp, len(gts) - len(matched)


def evaluate(detector, test_examples, label):
    yt, yp = [], []
    tp = fp = fn = 0
    ious = []
    empty_false_alarms = n_empty = 0
    for frame, gts in test_examples:
        preds = detector.predict(frame)
        yt.append(1 if gts else 0); yp.append(1 if preds else 0)
        t, f, n = box_level_counts(preds, gts)
        tp += t; fp += f; fn += n
        for g in gts:
            ious.append(max((iou_bbox(p.bbox, g.bbox) for p in preds), default=0.0))
        if not gts:
            n_empty += 1
            if preds:
                empty_false_alarms += 1
    cm = binary_confusion_matrix(yt, yp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    miou = sum(ious) / len(ious) if ious else 0.0
    print(f"\n=== {label} ===")
    print(f"  box-level : P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  (TP={tp} FP={fp} FN={fn})")
    print(f"  mean IoU  : {miou:.3f}")
    print(f"  presence  : acc={cm.accuracy:.3f} prec={cm.precision:.3f} "
          f"rec={cm.recall:.3f}  | empty frames: {n_empty}, "
          f"false alarms on empty: {empty_false_alarms}")


def main() -> None:
    idx = DatasetIndex("dataset", sensor_profile=MLX90640)
    empty = idx.find(EMPTY)

    # 1. Per-channel Tateno (even empty frames) + collect odd empty frames as negs.
    pre = {}
    empty_negs = []   # (raw_frame, []) — preprocessed below
    for ch in CHANNELS:
        calib = [empty.load_frame(ch, i) for i in range(empty.n_frames) if i % 2 == 0]
        pre[ch] = TatenoPipeline(MLX90640).fit(calib)
        for i in range(empty.n_frames):
            if i % 2 == 1:
                empty_negs.append((i, pre[ch].predict(empty.load_frame(ch, i))))
    print(f"Calibrated Tateno per channel on {EMPTY} (even frames); "
          f"{len(empty_negs)} empty negatives (odd frames).")

    # 2-3. Build + split.
    train, test = [], []
    for scene in SCENES:
        ds = FrameLevelDataset(idx, scenes=[scene], channels=CHANNELS,
                               class_filter=[PERSON_CLASS_ID], include_negative_frames=True)
        for key in ds._examples:
            frame, dets = ds._load_example(key)
            proc = pre[frame.camera_id].predict(frame)
            (test if key.frame_idx % TEST_EVERY == 0 else train).append((proc, dets))
    for i, proc in empty_negs:
        (test if i % TEST_EVERY == 0 else train).append((proc, []))

    n_pos = sum(1 for _, d in train if d)
    n_neg = sum(1 for _, d in train if not d)
    print(f"Examples: train={len(train)} (pos={n_pos}, empty={n_neg})  test={len(test)}")

    # 4. Train HOG+SVM.
    print("\nTraining HOGSVMDetector ...")
    hog = HOGSVMDetector(MLX90640, stride=(2, 2), score_threshold=0.0)
    hog.fit(train)
    print(f"  window={hog.window_size} cell={hog.cell_size} feat_dim={hog.feature_dim}")

    # 4b. Train MobileNet-SSD (deep).
    print("\nTraining MobileNetSSDDetector (deep, CPU) ...")
    ssd = MobileNetSSDDetector(MLX90640, n_epochs=25, batch_size=16, learning_rate=1e-3)
    ssd.fit(train, verbose=True)

    # 5. Evaluate all three.
    evaluate(hog, test, "HOG+SVM (trained, proc)")
    evaluate(ssd, test, "MobileNet-SSD (trained, proc)")
    adaptive = AdaptiveThresholdDetector(
        MLX90640, c_offset=0.4, min_solidity=0.45, min_area_pixels=4,
        pixel_value_bounds=(1.0, 25.0),
    ).fit([])
    evaluate(adaptive, test, "AdaptiveThreshold (no train, proc)")


if __name__ == "__main__":
    main()
