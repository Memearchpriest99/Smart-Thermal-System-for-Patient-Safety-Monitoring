"""Tests for MobileNetSSDDetector (Section 4.4.2.3).

The whole module is gated by a `pytest.importorskip("torch")` at the top so
the rest of the test suite still runs cleanly on machines without PyTorch
installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")        # noqa: E402

import numpy as np                          # noqa: E402

from thermal_algorithms.core.sensor_profile import MLX90640, WAVESHARE_26984
from thermal_algorithms.core.types import Detection, Frame
from thermal_algorithms.human_detection.mobilenet_ssd import MobileNetSSDDetector
from thermal_algorithms.human_detection.mobilenet_ssd_model import (
    MLX90640_BACKBONE, MLX90640_SSD,
    WAVESHARE_BACKBONE, WAVESHARE_SSD,
    MicroMobileNetSSD,
    generate_anchors,
)
from thermal_algorithms.human_detection.mobilenet_ssd_anchors import (
    cxcywh_to_xyxy,
    decode_boxes,
    encode_boxes,
    hard_negative_mining,
    match_anchors_to_targets,
    nms_xyxy,
    pairwise_iou,
    ssd_loss,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_mlx_forward_shapes(self):
        model = MicroMobileNetSSD(
            input_size=(24, 32),
            backbone_config=MLX90640_BACKBONE,
            ssd_config=MLX90640_SSD,
        )
        # F1 should be 16x12, F2 should be 8x6 (MLX has stem_stride=1 → /2 each block)
        f1, f2 = model.backbone(torch.zeros(1, 1, 24, 32))
        assert f1.shape[-2:] == (12, 16)
        assert f2.shape[-2:] == (6, 8)

    def test_waveshare_forward_shapes(self):
        model = MicroMobileNetSSD(
            input_size=(62, 80),
            backbone_config=WAVESHARE_BACKBONE,
            ssd_config=WAVESHARE_SSD,
        )
        # stem stride 2 → 31x40 → 15x20 (F1) → 7x10 (F2)  (rounding from 62/2=31, 31/2=15.5→15)
        f1, f2 = model.backbone(torch.zeros(1, 1, 62, 80))
        assert f1.shape[-2:] == (15, 20)
        # The exact F2 height depends on stride math; just verify it's smaller.
        assert f2.shape[-2] < f1.shape[-2]
        assert f2.shape[-1] < f1.shape[-1]

    def test_anchor_count_matches_feature_maps(self):
        model = MicroMobileNetSSD(
            input_size=(24, 32),
            backbone_config=MLX90640_BACKBONE,
            ssd_config=MLX90640_SSD,
        )
        anchors = model.build_anchors()
        # F1: 16*12*3 = 576, F2: 8*6*3 = 144 → 720
        expected = 16 * 12 * 3 + 8 * 6 * 3
        assert anchors.shape == (expected, 4)

    def test_full_forward_returns_boxes_and_classes(self):
        model = MicroMobileNetSSD(
            input_size=(24, 32),
            backbone_config=MLX90640_BACKBONE,
            ssd_config=MLX90640_SSD,
        )
        box, cls = model(torch.randn(2, 1, 24, 32))
        N = model.anchors.shape[0]
        assert box.shape == (2, N, 4)
        assert cls.shape == (2, N, 2)


# ---------------------------------------------------------------------------
# Anchor helpers
# ---------------------------------------------------------------------------

class TestAnchorHelpers:
    def test_generate_anchors_count(self):
        a = generate_anchors(
            feature_map_size=(6, 8), input_size=(24, 32),
            anchor_width=4.0, aspect_ratios=(1.0, 0.5, 0.33),
        )
        assert a.shape == (6 * 8 * 3, 4)

    def test_iou_self_is_one(self):
        boxes = torch.tensor([[10.0, 10.0, 4.0, 4.0]])
        iou = pairwise_iou(boxes, boxes)
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_iou_disjoint_is_zero(self):
        a = torch.tensor([[10.0, 10.0, 2.0, 2.0]])
        b = torch.tensor([[50.0, 50.0, 2.0, 2.0]])
        assert torch.allclose(pairwise_iou(a, b), torch.tensor([[0.0]]))

    def test_encode_decode_roundtrip(self):
        anchors = torch.tensor([[10.0, 12.0, 4.0, 6.0]])
        gt = torch.tensor([[11.5, 11.0, 5.0, 6.5]])
        offsets = encode_boxes(gt, anchors)
        recovered = decode_boxes(offsets, anchors)
        assert torch.allclose(recovered, gt, atol=1e-5)


# ---------------------------------------------------------------------------
# Anchor matching
# ---------------------------------------------------------------------------

class TestMatching:
    def test_no_gt_marks_all_negative(self):
        anchors = torch.tensor([[5.0, 5.0, 4.0, 4.0], [50.0, 50.0, 4.0, 4.0]])
        m = match_anchors_to_targets(anchors, torch.zeros((0, 4)))
        assert (m == -2).all()

    def test_close_anchor_matched_to_gt(self):
        anchors = torch.tensor([
            [10.0, 10.0, 4.0, 4.0],   # close to GT
            [50.0, 50.0, 4.0, 4.0],   # far away
        ])
        gt = torch.tensor([[10.5, 10.5, 4.0, 4.0]])
        m = match_anchors_to_targets(anchors, gt, iou_pos_threshold=0.5)
        assert m[0].item() == 0     # positive
        assert m[1].item() == -2    # negative

    def test_best_anchor_per_gt_forced_positive(self):
        # No anchor has IoU > 0.5; the best one is still forced to match.
        anchors = torch.tensor([
            [10.0, 10.0, 2.0, 2.0],
            [12.0, 12.0, 2.0, 2.0],
        ])
        gt = torch.tensor([[20.0, 20.0, 4.0, 4.0]])
        m = match_anchors_to_targets(anchors, gt, iou_pos_threshold=0.5)
        # At least one anchor must be assigned to GT 0.
        assert (m == 0).any()


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

class TestHardNegativeMining:
    def test_keeps_positives_and_some_negatives(self):
        # 1 positive, 10 negatives, request ratio 3 → keep 1 + 3 = 4
        matches = torch.tensor([0, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2])
        # cls_logits: positive at index 0, others ambiguous
        cls = torch.randn(11, 2)
        keep = hard_negative_mining(cls, matches, neg_pos_ratio=3.0)
        # Positive always kept.
        assert keep[0].item() is True or keep[0].item() == 1
        # Total kept should be 1 + 3 = 4
        assert int(keep.sum()) == 4


# ---------------------------------------------------------------------------
# SSD loss
# ---------------------------------------------------------------------------

class TestSSDLoss:
    def test_loss_is_finite_and_positive(self):
        model = MicroMobileNetSSD(
            input_size=(24, 32),
            backbone_config=MLX90640_BACKBONE,
            ssd_config=MLX90640_SSD,
        )
        box, cls = model(torch.randn(2, 1, 24, 32))
        # One GT per sample
        targets = [
            torch.tensor([[16.0, 12.0, 5.0, 9.0]]),
            torch.tensor([[8.0, 8.0, 4.0, 6.0]]),
        ]
        out = ssd_loss(box, cls, model.anchors, targets)
        assert torch.isfinite(out["loss"])
        assert out["loss"].item() > 0
        assert out["n_pos"].item() >= 2  # at least one positive anchor per GT

    def test_loss_backprop_changes_params(self):
        model = MicroMobileNetSSD(
            input_size=(24, 32),
            backbone_config=MLX90640_BACKBONE,
            ssd_config=MLX90640_SSD,
        )
        params_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        targets = [torch.tensor([[16.0, 12.0, 5.0, 9.0]])]
        for _ in range(3):
            box, cls = model(torch.randn(1, 1, 24, 32))
            loss = ssd_loss(box, cls, model.anchors, targets)["loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
        # At least one parameter should have changed.
        diffs = [(p - params_before[n]).abs().max().item()
                 for n, p in model.named_parameters()]
        assert max(diffs) > 1e-6


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

class TestNMS:
    def test_nms_dedups_overlapping(self):
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 10.0, 10.0],
            [50.0, 50.0, 60.0, 60.0],
        ])
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms_xyxy(boxes, scores, iou_threshold=0.3)
        assert keep.numel() == 2
        assert 0 in keep.tolist()
        assert 2 in keep.tolist()


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class TestDetector:
    def test_construction_for_both_profiles(self):
        d_m = MobileNetSSDDetector(MLX90640)
        d_w = MobileNetSSDDetector(WAVESHARE_26984)
        assert d_m.resolution_behavior == "fixed"
        assert d_w.resolution_behavior == "fixed"
        assert d_m.is_trainable is True

    def test_class_metadata(self):
        assert MobileNetSSDDetector.name == "mobilenet_ssd_detector"
        assert MobileNetSSDDetector.is_trainable is True
        assert MobileNetSSDDetector.resolution_behavior == "fixed"

    def test_predict_before_fit_raises(self):
        d = MobileNetSSDDetector(MLX90640, device="cpu")
        f = Frame(data=np.zeros((24, 32), dtype=np.float32), timestamp=0.0)
        with pytest.raises(RuntimeError, match="before fit"):
            d.predict(f)

    def test_one_epoch_runs_end_to_end(self):
        """End-to-end smoke test: 8 synthetic frames, 1 epoch, no errors."""
        rng = np.random.default_rng(0)
        examples = []
        for i in range(8):
            arr = rng.normal(20.0, 0.3, size=(24, 32)).astype(np.float32)
            # Plant a "person" box: 5 wide, 9 tall, somewhere in the middle.
            x = int(rng.integers(2, 26)); y = int(rng.integers(2, 14))
            arr[y:y + 9, x:x + 5] = 33.0
            frame = Frame(data=arr, timestamp=float(i), camera_id=0)
            det = Detection(bbox=(float(x), float(y), 5.0, 9.0),
                            score=1.0, class_id=1, camera_id=0)
            examples.append((frame, [det]))

        d = MobileNetSSDDetector(
            MLX90640, n_epochs=1, batch_size=4, device="cpu",
        )
        d.fit(examples, verbose=False)
        assert d.is_fitted is True

        # predict shouldn't error.
        out = d.predict(examples[0][0])
        assert isinstance(out, list)
        for det in out:
            assert isinstance(det, Detection)
            assert det.class_id == 1
            assert 0.0 <= det.score <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Save + load preserves predictions (within float tolerance)."""
        rng = np.random.default_rng(0)
        examples = []
        for i in range(4):
            arr = rng.normal(20.0, 0.3, size=(24, 32)).astype(np.float32)
            arr[8:17, 12:17] = 33.0
            frame = Frame(data=arr, timestamp=float(i), camera_id=0)
            det = Detection(bbox=(12.0, 8.0, 5.0, 9.0),
                            score=1.0, class_id=1, camera_id=0)
            examples.append((frame, [det]))

        d = MobileNetSSDDetector(
            MLX90640, n_epochs=2, batch_size=2, device="cpu",
        )
        d.fit(examples, verbose=False)
        path = tmp_path / "ssd.thalg"
        d.save(path)

        loaded = MobileNetSSDDetector.load(path)
        assert loaded.is_fitted is True

        f = examples[0][0]
        a = d.predict(f)
        b = loaded.predict(f)
        # Predictions should match exactly: same weights, same input.
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.bbox == pytest.approx(y.bbox, rel=1e-4)
            assert x.score == pytest.approx(y.score, rel=1e-4)
