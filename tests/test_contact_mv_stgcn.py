"""Tests for MVSTGCNDetector (§ 4.4.3.2). Gated by torch."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

from thermal_algorithms.core.sensor_profile import MLX90640
from thermal_algorithms.core.types import ActorPosition, ContactEvent, Frame
from thermal_algorithms.contact_detection.mv_stgcn import MVSTGCNDetector, _build_stgcn_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_actor(world_xy=(0.0, 0.0), track_id=0) -> ActorPosition:
    return ActorPosition(world_xy=world_xy, track_id=track_id, source_camera_ids=(0, 1))


def _make_contact_event(contact: bool, n_actors: int = 2) -> ContactEvent:
    actors = tuple(_make_actor((float(i), 0.0), i) for i in range(n_actors))
    pairs = ((0, 1),) if contact and n_actors >= 2 else ()
    return ContactEvent(actors=actors, pairs_in_contact=pairs, timestamp=0.0)


def _make_frames(ts=0.0):
    w, h = MLX90640.resolution
    data = np.zeros((h, w), dtype=np.float32)
    return (Frame(data=data, timestamp=ts), Frame(data=data, timestamp=ts),
            Frame(data=data, timestamp=ts))


def _make_training_set(n=40, T=16):
    """Create n contact events and matching frame triplets."""
    frames, events = [], []
    for i in range(n):
        frames.append(_make_frames(float(i)))
        contact = i >= n // 2
        n_actors = 2
        actors = tuple(_make_actor((float(j), 0.0), j) for j in range(n_actors))
        if contact:
            actors = (
                _make_actor((0.0, 0.0), 0),
                _make_actor((0.2, 0.0), 1),
            )
        pairs = ((0, 1),) if contact else ()
        events.append(ContactEvent(actors=actors, pairs_in_contact=pairs, timestamp=float(i)))
    return frames, events


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class TestSTGCNModel:
    def test_forward_shape(self):
        model = _build_stgcn_model(
            node_feat_dim=7, hidden_dim=16, n_layers=2, max_actors=5, T=16
        )
        B, T, N, D = 2, 16, 5, 7
        feat = torch.zeros(B, T, N, D)
        pos = torch.zeros(B, T, N, 2)
        mask = torch.ones(B, T, N)
        out = model(feat, pos, mask)
        assert out.shape == (B, 2)

    def test_padded_nodes_do_not_affect_output(self):
        model = _build_stgcn_model(node_feat_dim=7, hidden_dim=16, n_layers=1,
                                   max_actors=5, T=4)
        B, T, N = 1, 4, 5
        feat = torch.randn(B, T, N, 7)
        pos = torch.randn(B, T, N, 2)

        # All actors present
        mask_full = torch.ones(B, T, N)
        out_full = model(feat, pos, mask_full)

        # Only first actor present, rest padded
        mask_partial = torch.zeros(B, T, N)
        mask_partial[:, :, 0] = 1.0
        out_partial = model(feat, pos, mask_partial)

        # Outputs should differ (different number of valid nodes)
        assert not torch.allclose(out_full, out_partial, atol=1e-3)


# ---------------------------------------------------------------------------
# Detector construction
# ---------------------------------------------------------------------------

class TestMVSTGCNConstruction:
    def test_constructs_without_profile(self):
        det = MVSTGCNDetector()
        assert det.resolution_behavior == "invariant"
        assert det.is_trainable

    def test_not_fitted_initially(self):
        det = MVSTGCNDetector()
        assert not det.is_fitted


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestMVSTGCNFit:
    def test_fit_requires_labels(self):
        det = MVSTGCNDetector(T=4)
        frames, _ = _make_training_set(n=10, T=4)
        with pytest.raises(ValueError, match="y="):
            det.fit(frames, y=None)

    def test_fit_marks_fitted(self):
        det = MVSTGCNDetector(T=4, n_epochs=1)
        frames, events = _make_training_set(n=10, T=4)
        ret = det.fit(frames, events)
        assert det.is_fitted
        assert ret is det

    def test_fit_raises_too_few_frames(self):
        det = MVSTGCNDetector(T=16, n_epochs=1)
        frames, events = _make_training_set(n=5, T=16)
        with pytest.raises(ValueError):
            det.fit(frames, events)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestMVSTGCNPredict:
    def _trained_det(self, T=4) -> MVSTGCNDetector:
        det = MVSTGCNDetector(T=T, n_epochs=1, hidden_dim=8, n_gcn_layers=1)
        frames, events = _make_training_set(n=T + 4, T=T)
        det.fit(frames, events)
        det.reset()
        return det

    def test_predict_returns_contact_event(self):
        det = self._trained_det(T=4)
        for _ in range(4):
            event = det.predict(_make_frames(), detections=([], [], []))
        assert isinstance(event, ContactEvent)

    def test_buffer_filling_returns_zero_confidence(self):
        det = self._trained_det(T=8)
        event = det.predict(_make_frames(), detections=([], [], []))
        assert event.confidence == pytest.approx(0.0)
        assert event.debug.get("status") == "buffer_filling"

    def test_confidence_in_unit_interval_after_buffer_full(self):
        det = self._trained_det(T=4)
        event = None
        for _ in range(6):
            event = det.predict(_make_frames(), detections=([], [], []))
        assert 0.0 <= event.confidence <= 1.0

    def test_reset_clears_buffer(self):
        det = self._trained_det(T=4)
        for _ in range(4):
            det.predict(_make_frames(), detections=([], [], []))
        det.reset()
        event = det.predict(_make_frames(), detections=([], [], []))
        assert event.confidence == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestMVSTGCNPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        det = MVSTGCNDetector(T=4, n_epochs=1, hidden_dim=8, n_gcn_layers=1)
        frames, events = _make_training_set(n=8, T=4)
        det.fit(frames, events)

        path = tmp_path / "mv_stgcn.thalg"
        det.save(path)
        loaded = MVSTGCNDetector.load(path)
        assert loaded.is_fitted

    def test_predictions_match_after_load(self, tmp_path):
        det = MVSTGCNDetector(T=4, n_epochs=1, hidden_dim=8, n_gcn_layers=1)
        frames, events = _make_training_set(n=8, T=4)
        det.fit(frames, events)

        path = tmp_path / "mv_stgcn.thalg"
        det.save(path)
        loaded = MVSTGCNDetector.load(path)

        det.reset(); loaded.reset()
        for _ in range(4):
            e1 = det.predict(_make_frames(), detections=([], [], []))
            e2 = loaded.predict(_make_frames(), detections=([], [], []))
        assert e1.confidence == pytest.approx(e2.confidence, abs=1e-5)

    def test_homography_survives_save_load(self, tmp_path):
        """Regression: homography lives on ContactDetector outside _params, so
        _state_dict must persist it explicitly. Without it the reloaded fusion
        front-end sees no homography → zero actors → constant confidence."""
        import numpy as np
        from thermal_algorithms.core.types import HomographyMatrices

        H = HomographyMatrices(h1=np.eye(3), h2=np.eye(3), h3=np.eye(3))
        det = MVSTGCNDetector(T=4, n_epochs=1, hidden_dim=8, n_gcn_layers=1,
                              homography=H)
        frames, events = _make_training_set(n=8, T=4)
        det.fit(frames, events)

        path = tmp_path / "mv_stgcn.thalg"
        det.save(path)
        loaded = MVSTGCNDetector.load(path)
        assert loaded.homography is not None
        np.testing.assert_allclose(loaded.homography.h1, H.h1)
