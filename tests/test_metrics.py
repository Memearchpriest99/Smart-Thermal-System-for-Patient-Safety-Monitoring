"""Tests for training/metrics.py.

Where possible, tests use numbers from the Engineering Report directly:
  - §5.3.2 Table 3: Human Detection confusion matrices
  - §5.3.4 Table 4: Fire Detection confusion matrices + IoU
  - §5.3.1: SBR 5.61× improvement
"""

from __future__ import annotations

import numpy as np
import pytest

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


# ===========================================================================
# BinaryConfusionMatrix construction and derived metrics
# ===========================================================================

class TestBinaryConfusionMatrix:
    def test_construct_directly(self):
        cm = BinaryConfusionMatrix(tp=220, tn=62, fp=5, fn=29)
        assert cm.tp == 220
        assert cm.total == 316

    def test_accuracy_matches_report_processed(self):
        # §5.3.2 "1_man_out_in_out_7_sec", Processed: 60/66 = 90.9%
        cm = BinaryConfusionMatrix(tp=48, tn=12, fp=0, fn=6)
        # Exact numbers not given, use proportional: acc=90.9%, total=66
        # Just verify the formula works
        cm2 = BinaryConfusionMatrix(tp=53, tn=7, fp=0, fn=6)
        assert cm2.accuracy == pytest.approx((53 + 7) / 66)

    def test_precision_formula(self):
        cm = BinaryConfusionMatrix(tp=10, tn=5, fp=2, fn=3)
        assert cm.precision == pytest.approx(10 / 12)

    def test_recall_formula(self):
        cm = BinaryConfusionMatrix(tp=10, tn=5, fp=2, fn=3)
        assert cm.recall == pytest.approx(10 / 13)

    def test_f1_formula(self):
        cm = BinaryConfusionMatrix(tp=10, tn=5, fp=2, fn=3)
        p, r = cm.precision, cm.recall
        assert cm.f1 == pytest.approx(2 * p * r / (p + r))

    def test_f1_perfect(self):
        cm = BinaryConfusionMatrix(tp=100, tn=50, fp=0, fn=0)
        assert cm.f1 == pytest.approx(1.0)
        assert cm.recall == pytest.approx(1.0)
        assert cm.precision == pytest.approx(1.0)

    def test_zero_tp_gives_zero_precision_recall_f1(self):
        cm = BinaryConfusionMatrix(tp=0, tn=10, fp=5, fn=3)
        assert cm.precision == 0.0
        assert cm.recall == 0.0
        assert cm.f1 == 0.0

    def test_false_alarm_rate(self):
        # FAR = FP / (FP + TN)
        cm = BinaryConfusionMatrix(tp=0, tn=36, fp=31, fn=0)
        assert cm.false_alarm_rate == pytest.approx(31 / 67)

    def test_correct_over_total_property(self):
        cm = BinaryConfusionMatrix(tp=163, tn=36, fp=31, fn=86)
        assert cm.correct == 199
        assert cm.total == 316

    def test_add_combines_cells(self):
        a = BinaryConfusionMatrix(tp=10, tn=5, fp=2, fn=3)
        b = BinaryConfusionMatrix(tp=20, tn=10, fp=1, fn=1)
        c = a + b
        assert c.tp == 30
        assert c.tn == 15
        assert c.fp == 3
        assert c.fn == 4

    def test_as_matrix_shape(self):
        cm = BinaryConfusionMatrix(tp=5, tn=3, fp=1, fn=2)
        m = cm.as_matrix()
        assert m.shape == (2, 2)
        assert m[0, 0] == 3   # TN
        assert m[1, 1] == 5   # TP


# Report numbers: §5.3.2 raw confusion matrix (Figure 23)
# Raw baseline: TP=163, TN=36, FP=31, FN=86
class TestReportHumanDetectionRaw:
    """Verify exact metric values from §5.3.2 Table 3 / Figure 23 (raw data)."""

    def setup_method(self):
        # Figure 23 raw matrix: Empty row [36 FP=31], Human row [FN=86 TP=163]
        self.cm = BinaryConfusionMatrix(tp=163, tn=36, fp=31, fn=86)

    def test_total(self):
        assert self.cm.total == 316

    def test_precision_raw(self):
        # TP / (TP + FP) = 163 / (163 + 31) = 163/194
        assert self.cm.precision == pytest.approx(163 / 194, abs=0.001)

    def test_recall_raw(self):
        # TP / (TP + FN) = 163 / (163 + 86) = 163/249
        assert self.cm.recall == pytest.approx(163 / 249, abs=0.001)


class TestReportFireDetectionProcessed:
    """Verify §5.3.4 aggregate processed metrics: Recall=1.0, Precision=0.8015."""

    def test_recall_one_zero(self):
        # Optimised pipeline: 0 False Negatives → Recall = 1.0
        cm = BinaryConfusionMatrix(tp=315, tn=66, fp=78, fn=0)
        assert cm.recall == pytest.approx(1.0)

    def test_precision_approx(self):
        # Precision ≈ 0.8015
        cm = BinaryConfusionMatrix(tp=315, tn=66, fp=78, fn=0)
        # 315 / (315 + 78) = 315 / 393 ≈ 0.8015
        assert cm.precision == pytest.approx(315 / 393, abs=0.001)


# ===========================================================================
# binary_confusion_matrix builder
# ===========================================================================

class TestBinaryConfusionMatrixBuilder:
    def test_all_correct(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 1, 0, 0, 1]
        cm = binary_confusion_matrix(y_true, y_pred)
        assert cm.tp == 3 and cm.tn == 2 and cm.fp == 0 and cm.fn == 0

    def test_all_wrong(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        cm = binary_confusion_matrix(y_true, y_pred)
        assert cm.tp == 0 and cm.tn == 0 and cm.fp == 2 and cm.fn == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            binary_confusion_matrix([1, 0], [1])

    def test_numpy_arrays_accepted(self):
        cm = binary_confusion_matrix(np.array([1, 0, 1]), np.array([1, 1, 1]))
        assert cm.fp == 1


class TestDetectionsToBinaryLabels:
    def test_empty_detection_is_negative(self):
        y_pred, y_true = detections_to_binary_labels([[]], [[]])
        assert y_pred == [0]
        assert y_true == [0]

    def test_non_empty_detection_is_positive(self):
        y_pred, y_true = detections_to_binary_labels([["det1"]], [["gt1"]])
        assert y_pred == [1]
        assert y_true == [1]

    def test_false_negative_frame(self):
        # GT present, detector returns nothing
        y_pred, y_true = detections_to_binary_labels([[]], [["gt1"]])
        assert y_pred == [0]
        assert y_true == [1]

    def test_false_positive_frame(self):
        # GT empty, detector fires
        y_pred, y_true = detections_to_binary_labels([["det1"]], [[]])
        assert y_pred == [1]
        assert y_true == [0]


# ===========================================================================
# IoU metrics
# ===========================================================================

class TestIoUBBox:
    def test_perfect_overlap(self):
        assert iou_bbox((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou_bbox((0, 0, 5, 5), (10, 10, 5, 5)) == pytest.approx(0.0)

    def test_half_overlap(self):
        # pred covers (0,0)→(10,10), gt covers (5,0)→(15,10)
        # intersection = 5×10=50, union = 100+100-50=150
        assert iou_bbox((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_contained_box(self):
        # gt fully inside pred: inter=25, union=100+25-25=100
        assert iou_bbox((0, 0, 10, 10), (2, 2, 5, 5)) == pytest.approx(25 / 100)

    def test_zero_area_returns_zero(self):
        assert iou_bbox((0, 0, 0, 5), (0, 0, 5, 5)) == pytest.approx(0.0)

    def test_symmetric(self):
        a = (1.0, 2.0, 5.0, 3.0)
        b = (3.0, 1.0, 4.0, 6.0)
        assert iou_bbox(a, b) == pytest.approx(iou_bbox(b, a))


class TestIoUMaskVsBBox:
    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        assert iou_mask_vs_bbox(mask, (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:5, 0:5] = 1
        assert iou_mask_vs_bbox(mask, (6, 6, 4, 4)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:5, 0:10] = 1            # top half
        # gt_bbox covers the whole frame
        iou = iou_mask_vs_bbox(mask, (0, 0, 10, 10))
        # inter = 50, union = 100 → 0.5
        assert iou == pytest.approx(0.5)

    def test_empty_mask_returns_zero(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        assert iou_mask_vs_bbox(mask, (0, 0, 5, 5)) == pytest.approx(0.0)

    def test_degenerate_bbox_returns_zero(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        assert iou_mask_vs_bbox(mask, (0, 0, 0, 5)) == pytest.approx(0.0)


class TestMeanDetectionIoU:
    def test_empty_gt_frames_excluded(self):
        # Frame 0: no GT → excluded; Frame 1: GT present → included
        preds = [[(0, 0, 5, 5)], [(0, 0, 5, 5)]]
        gts = [[], [(0, 0, 5, 5)]]
        assert mean_detection_iou(preds, gts) == pytest.approx(1.0)

    def test_all_empty_returns_zero(self):
        assert mean_detection_iou([[], []], [[], []]) == pytest.approx(0.0)

    def test_no_predictions_for_positive_frame_is_zero(self):
        preds = [[]]
        gts = [[(0, 0, 5, 5)]]
        assert mean_detection_iou(preds, gts) == pytest.approx(0.0)


# ===========================================================================
# Signal-to-Background Ratio
# ===========================================================================

class TestSBR:
    def test_hot_region_higher_sbr(self):
        frame = np.full((24, 32), 25.0, dtype=np.float32)
        frame[8:16, 12:20] = 35.0     # hot blob
        gt_bbox = (12, 8, 8, 8)
        sbr = signal_to_background_ratio(frame, [gt_bbox])
        # signal mean = 35, background mean = 25 → SBR ≈ 1.4
        assert sbr == pytest.approx(35.0 / 25.0, abs=0.01)

    def test_uniform_frame_sbr_one(self):
        frame = np.full((24, 32), 25.0, dtype=np.float32)
        gt_bbox = (0, 0, 16, 12)
        sbr = signal_to_background_ratio(frame, [gt_bbox])
        assert sbr == pytest.approx(1.0)

    def test_no_gt_bbox_returns_one(self):
        frame = np.random.default_rng(0).random((24, 32)).astype(np.float32)
        assert signal_to_background_ratio(frame, []) == pytest.approx(1.0)

    def test_preprocessing_improves_sbr(self):
        # Simulate raw (noisy) vs processed (clean background subtracted).
        rng = np.random.default_rng(42)
        h, w = 24, 32
        # Raw: warm background + person blob
        raw = (rng.standard_normal((h, w)) * 2 + 25).astype(np.float32)
        raw[8:16, 12:20] = 35.0
        # Processed (background subtracted): background ≈ 0, person signal preserved
        processed = np.abs(raw - 25.0).astype(np.float32)

        gt_bbox = (12, 8, 8, 8)
        sbr_raw = signal_to_background_ratio(raw, [gt_bbox])
        sbr_proc = signal_to_background_ratio(processed, [gt_bbox])
        assert sbr_proc > sbr_raw

    def test_mean_sbr_averages_across_frames(self):
        frame1 = np.full((24, 32), 25.0, dtype=np.float32)
        frame1[10:14, 14:18] = 35.0
        frame2 = np.full((24, 32), 25.0, dtype=np.float32)
        frame2[10:14, 14:18] = 45.0

        gt = [(14, 10, 4, 4)]
        sbr1 = signal_to_background_ratio(frame1, gt)
        sbr2 = signal_to_background_ratio(frame2, gt)
        expected = (sbr1 + sbr2) / 2
        assert mean_sbr([frame1, frame2], [gt, gt]) == pytest.approx(expected)


# ===========================================================================
# ScenarioResult and table formatting
# ===========================================================================

class TestScenarioResult:
    def _make_result(self, name="test_scene", mode="proc") -> ScenarioResult:
        return ScenarioResult(
            name=name,
            mode=mode,
            confusion=BinaryConfusionMatrix(tp=163, tn=36, fp=31, fn=86),
            mean_iou=0.5387,
        )

    def test_accuracy_delegated(self):
        r = self._make_result()
        assert r.accuracy == r.confusion.accuracy

    def test_correct_over_total_string(self):
        r = self._make_result()
        assert r.correct_over_total == "199/316"

    def test_format_table_contains_header(self):
        r = self._make_result()
        table = format_scenario_table([r])
        assert "Accuracy" in table
        assert "Precision" in table
        assert "Recall" in table
        assert "F1" in table

    def test_format_table_includes_iou_when_requested(self):
        r = self._make_result()
        table_with = format_scenario_table([r], include_iou=True)
        table_without = format_scenario_table([r], include_iou=False)
        assert "IoU" in table_with
        assert "IoU" not in table_without

    def test_format_table_contains_scenario_name(self):
        r = self._make_result(name="2_men_far_near_far")
        table = format_scenario_table([r])
        assert "2_men_far_near_far" in table


class TestAggregateScenarios:
    def _results(self):
        return [
            ScenarioResult("s1", "raw", BinaryConfusionMatrix(tp=10, tn=5, fp=2, fn=3)),
            ScenarioResult("s1", "proc", BinaryConfusionMatrix(tp=15, tn=7, fp=1, fn=1)),
            ScenarioResult("s2", "raw", BinaryConfusionMatrix(tp=8, tn=4, fp=3, fn=5)),
        ]

    def test_all_modes(self):
        agg = aggregate_scenarios(self._results())
        assert agg.tp == 33
        assert agg.total == 64

    def test_filter_by_mode(self):
        agg = aggregate_scenarios(self._results(), mode="raw")
        assert agg.tp == 18   # 10 + 8
        assert agg.fp == 5    # 2 + 3

    def test_empty_results(self):
        agg = aggregate_scenarios([])
        assert agg.total == 0
        assert agg.accuracy == 0.0
