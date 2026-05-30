"""Tests for multi_view utilities: homography, tracker, fusion."""

from __future__ import annotations

import math
import numpy as np
import pytest

from thermal_algorithms.core.types import Detection, Frame, HomographyMatrices
from thermal_algorithms.contact_detection.multi_view.homography import (
    project_foot_point,
    solve_homography_from_markers,
    _dlt_solve,
)
from thermal_algorithms.contact_detection.multi_view.tracker import (
    KalmanTrack,
    PerCameraTracker,
)
from thermal_algorithms.contact_detection.multi_view.fusion import fuse_detections


# ---------------------------------------------------------------------------
# Homography
# ---------------------------------------------------------------------------

def _make_synthetic_H() -> np.ndarray:
    """A known 3×3 homography for testing (roughly: scale + translate)."""
    H = np.array([
        [0.1,  0.0,  1.0],
        [0.0,  0.1,  0.5],
        [0.0,  0.0,  1.0],
    ], dtype=np.float64)
    return H


def _make_correspondences_from_H(
    H: np.ndarray,
    n_points: int = 8,
    camera_id: int = 0,
) -> tuple[int, list[tuple[tuple[float, float], tuple[float, float]]]]:
    """Generate synthetic (u,v)↔(Xw,Yw) pairs from a known H."""
    rng = np.random.default_rng(42)
    pairs = []
    for _ in range(n_points):
        u, v = rng.uniform(0, 32), rng.uniform(0, 24)
        p = H @ np.array([u, v, 1.0])
        xw, yw = p[0] / p[2], p[1] / p[2]
        pairs.append(((float(u), float(v)), (float(xw), float(yw))))
    return (camera_id, pairs)


class TestHomography:
    def test_dlt_recovers_known_H(self):
        H_true = _make_synthetic_H()
        _, pairs = _make_correspondences_from_H(H_true, n_points=8)
        H_est = _dlt_solve(pairs)
        # Ratio should be ~1 for all non-zero elements
        np.testing.assert_allclose(H_est / H_est[2, 2], H_true / H_true[2, 2], atol=1e-6)

    def test_solve_requires_4_points(self):
        H_true = _make_synthetic_H()
        _, pairs = _make_correspondences_from_H(H_true, n_points=3)
        with pytest.raises(ValueError, match="at least 4"):
            solve_homography_from_markers([(0, pairs)])

    def test_solve_returns_3x3_matrices(self):
        H_true = _make_synthetic_H()
        corr = [_make_correspondences_from_H(H_true, camera_id=k) for k in range(3)]
        result = solve_homography_from_markers(corr)
        assert result.h1.shape == (3, 3)
        assert result.h2.shape == (3, 3)
        assert result.h3.shape == (3, 3)

    def test_missing_cameras_get_fallback(self):
        H_true = _make_synthetic_H()
        corr = [_make_correspondences_from_H(H_true, camera_id=0)]
        result = solve_homography_from_markers(corr)
        # h2 and h3 should be identity (fallback)
        np.testing.assert_array_almost_equal(result.h2, np.eye(3))
        np.testing.assert_array_almost_equal(result.h3, np.eye(3))

    def test_project_foot_point_recovers_world_coords(self):
        H = _make_synthetic_H()
        u, v = 16.0, 12.0
        p = H @ np.array([u, v, 1.0])
        xw_expected, yw_expected = p[0] / p[2], p[1] / p[2]
        xw, yw = project_foot_point((u, v), H)
        assert abs(xw - xw_expected) < 1e-6
        assert abs(yw - yw_expected) < 1e-6

    def test_project_degenerate_returns_zero(self):
        H = np.zeros((3, 3))
        xw, yw = project_foot_point((5.0, 5.0), H)
        assert xw == 0.0
        assert yw == 0.0


# ---------------------------------------------------------------------------
# Kalman Tracker
# ---------------------------------------------------------------------------

class TestKalmanTrack:
    def test_predict_advances_state(self):
        track = KalmanTrack(
            track_id=0,
            state=np.array([10.0, 20.0, 2.0, -1.0]),
            cov=np.eye(4),
        )
        from thermal_algorithms.contact_detection.multi_view.tracker import _process_noise
        Q = _process_noise(1.0)
        track.predict(dt=0.125, Q=Q)
        # u should advance by 2.0 * 0.125 = 0.25
        assert abs(track.state[0] - 10.25) < 1e-6
        assert abs(track.state[1] - (20.0 - 0.125)) < 1e-6

    def test_update_moves_toward_measurement(self):
        track = KalmanTrack(
            track_id=0,
            state=np.array([10.0, 10.0, 0.0, 0.0]),
            cov=np.eye(4) * 4.0,
        )
        R = np.diag([1.5 ** 2, 1.5 ** 2])
        # Measurement at (12, 10) — track should move toward it
        track.update(np.array([12.0, 10.0]), R)
        assert track.state[0] > 10.0
        assert track.misses == 0

    def test_misses_set_to_zero_after_update(self):
        track = KalmanTrack(
            track_id=1,
            state=np.array([5.0, 5.0, 0.0, 0.0]),
            cov=np.eye(4),
        )
        track.misses = 3
        R = np.diag([1.0, 1.0])
        track.update(np.array([5.0, 5.0]), R)
        assert track.misses == 0


class TestPerCameraTracker:
    def test_creates_track_for_new_detection(self):
        tracker = PerCameraTracker()
        tracker.update([(10.0, 15.0)])
        assert len(tracker.tracks) == 1

    def test_updates_existing_track(self):
        tracker = PerCameraTracker()
        tracker.update([(10.0, 15.0)])
        tid0 = list(tracker.tracks.keys())[0]
        tracker.predict_all()
        tracker.update([(10.5, 15.0)])
        assert tid0 in tracker.tracks  # same track, not a new one

    def test_prunes_stale_tracks(self):
        tracker = PerCameraTracker(max_age=2)
        tracker.update([(5.0, 5.0)])
        # Miss 3 consecutive frames
        for _ in range(3):
            tracker.predict_all()
            tracker.update([])
        assert len(tracker.tracks) == 0

    def test_births_new_track_for_far_detection(self):
        tracker = PerCameraTracker(max_assoc_dist=5.0)
        tracker.update([(5.0, 5.0)])
        # Detection 30 px away → cannot be associated → new track
        tracker.predict_all()
        tracker.update([(5.0, 5.0), (35.0, 5.0)])
        assert len(tracker.tracks) == 2

    def test_reset_clears_all_tracks(self):
        tracker = PerCameraTracker()
        tracker.update([(1.0, 2.0), (10.0, 11.0)])
        tracker.reset()
        assert len(tracker.tracks) == 0


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _det(world_uv: tuple[float, float], camera_id: int = 0) -> Detection:
    """Stub Detection whose foot_point is exactly world_uv (identity H)."""
    u, v = world_uv
    # bbox such that foot_point = (x + w/2, y + h) = (u, v)
    # e.g. x=u-1, w=2, y=v-2, h=2 → foot=(u, v)
    return Detection(bbox=(u - 1.0, v - 2.0, 2.0, 2.0), score=1.0, camera_id=camera_id)


def _identity_homographies() -> HomographyMatrices:
    return HomographyMatrices(
        h1=np.eye(3), h2=np.eye(3), h3=np.eye(3),
    )


class TestFusion:
    def test_single_source_discarded(self):
        dets = [[_det((1.0, 1.0), camera_id=0)], [], []]
        actors, _ = fuse_detections(dets, _identity_homographies())
        assert actors == []

    def test_two_consistent_sources_produce_one_actor(self):
        # Both cameras see the subject at (1.0, 1.0) — should cluster and validate
        dets = [
            [_det((1.0, 1.0), camera_id=0)],
            [_det((1.0, 1.0), camera_id=1)],
            [],
        ]
        actors, _ = fuse_detections(dets, _identity_homographies(), epsilon_m=0.5)
        assert len(actors) == 1
        assert abs(actors[0].world_xy[0] - 1.0) < 1e-3
        assert abs(actors[0].world_xy[1] - 1.0) < 1e-3

    def test_two_actors_distinct_clusters(self):
        dets = [
            [_det((1.0, 1.0), camera_id=0), _det((5.0, 5.0), camera_id=0)],
            [_det((1.05, 1.0), camera_id=1), _det((5.05, 5.0), camera_id=1)],
            [],
        ]
        actors, _ = fuse_detections(dets, _identity_homographies(), epsilon_m=0.5)
        assert len(actors) == 2

    def test_n3_outlier_removal(self):
        # Two cameras consistent (0.1 m apart), third camera outlier (2 m away)
        dets = [
            [_det((0.0, 0.0), camera_id=0)],
            [_det((0.1, 0.0), camera_id=1)],
            [_det((2.0, 0.0), camera_id=2)],  # outlier
        ]
        actors, _ = fuse_detections(dets, _identity_homographies(), epsilon_m=0.5)
        # Outlier from cam2 removed; cam0+cam1 cluster survives
        assert len(actors) == 1

    def test_empty_detections(self):
        actors, _ = fuse_detections([[], [], []], _identity_homographies())
        assert actors == []

    def test_track_ids_increment(self):
        dets = [
            [_det((0.0, 0.0), camera_id=0)],
            [_det((0.05, 0.0), camera_id=1)],
            [],
        ]
        actors1, nxt = fuse_detections(dets, _identity_homographies(), next_track_id=10)
        actors2, _ = fuse_detections(dets, _identity_homographies(), next_track_id=nxt)
        assert actors1[0].track_id == 10
        assert actors2[0].track_id == 11
