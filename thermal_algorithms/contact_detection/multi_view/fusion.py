"""Foot-point projection and multi-view fusion to unique actor nodes (§ 4.4.3).

Pipeline
--------
1. For each camera k, project every detection's foot-point (bottom-centre of
   its bounding box) through H_k to obtain a 2-D world coordinate on the
   floor plane (Z = 0).

2. Validate the projected cloud — discard single-source detections:
   - N = 1 projections have no cross-camera corroboration → discard.
   - N = 3 projections run outlier removal: if one point is inconsistent
     with both others (its distances to both peers exceed ε), discard it.
   - N ≥ 2 consistent projections pass forward.

3. Cluster the validated projections within ε metres using a greedy
   single-linkage approach.  Each cluster's centroid becomes one
   ``ActorPosition`` node.

The resulting list of ``ActorPosition`` objects has one entry per unique
person visible in the scene at this timestep.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from thermal_algorithms.core.types import ActorPosition, Detection, HomographyMatrices
from thermal_algorithms.contact_detection.multi_view.homography import project_foot_point


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_detections(
    detections_per_camera: list[list[Detection]],
    homographies: HomographyMatrices,
    *,
    epsilon_m: float = 0.5,
    next_track_id: int = 0,
) -> tuple[list[ActorPosition], int]:
    """Project and fuse detections from 3 cameras into unique actor positions.

    Args:
        detections_per_camera: A 3-element list: ``[dets_cam0, dets_cam1,
            dets_cam2]``.  Each element is the output of a ``HumanDetector``
            for that camera.
        homographies: The three calibrated homography matrices.
        epsilon_m: Cross-camera consistency threshold in metres.  Projected
            foot-points within this distance of one another are considered to
            represent the same person.
        next_track_id: Starting value for IDs assigned to new ActorPosition
            nodes.  Pass a counter to maintain globally unique IDs.

    Returns:
        ``(actor_list, updated_next_track_id)``
    """
    # Step 1 — project all foot-points to world coordinates
    world_points: list[_WorldPoint] = []
    for cam_id, dets in enumerate(detections_per_camera):
        H = homographies[cam_id]
        for det in dets:
            xw, yw = project_foot_point(det.foot_point, H)
            confidence = det.score
            world_points.append(_WorldPoint(
                world_xy=(xw, yw),
                camera_id=cam_id,
                confidence=confidence,
            ))

    if not world_points:
        return [], next_track_id

    # Step 2 — cluster into candidate groups
    clusters = _greedy_cluster(world_points, epsilon_m)

    # Step 3 — validate each cluster and compute centroid
    actors: list[ActorPosition] = []
    for cluster in clusters:
        validated = _validate_cluster(cluster, epsilon_m)
        if not validated:
            continue
        xs = [p.world_xy[0] for p in validated]
        ys = [p.world_xy[1] for p in validated]
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        cam_ids = tuple(sorted({p.camera_id for p in validated}))
        conf = float(np.mean([p.confidence for p in validated]))
        actors.append(ActorPosition(
            world_xy=(cx, cy),
            track_id=next_track_id,
            source_camera_ids=cam_ids,
            confidence=conf,
        ))
        next_track_id += 1

    return actors, next_track_id


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class _WorldPoint:
    __slots__ = ("world_xy", "camera_id", "confidence")

    def __init__(
        self,
        world_xy: tuple[float, float],
        camera_id: int,
        confidence: float,
    ) -> None:
        self.world_xy = world_xy
        self.camera_id = camera_id
        self.confidence = confidence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _greedy_cluster(
    points: list[_WorldPoint],
    epsilon_m: float,
) -> list[list[_WorldPoint]]:
    """Single-linkage greedy clustering by Euclidean distance.

    Each point is assigned to the first existing cluster that has any member
    within ``epsilon_m`` of it; otherwise a new cluster is started.
    """
    clusters: list[list[_WorldPoint]] = []
    for pt in points:
        assigned = False
        for cluster in clusters:
            for member in cluster:
                d = _dist(pt.world_xy, member.world_xy)
                if d < epsilon_m:
                    cluster.append(pt)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            clusters.append([pt])
    return clusters


def _validate_cluster(
    cluster: list[_WorldPoint],
    epsilon_m: float,
) -> list[_WorldPoint]:
    """Apply the § 4.4.3 validation rules.

    - N = 1: discard (no cross-camera corroboration).
    - N = 2: accept if pairwise distance < ε.
    - N = 3: outlier removal then accept ≥ 2 consistent points.
    - N > 3: keep all points (rare; already within epsilon due to clustering).
    """
    n = len(cluster)
    if n == 1:
        return []                # single-source → discard

    if n == 2:
        d = _dist(cluster[0].world_xy, cluster[1].world_xy)
        return cluster if d < epsilon_m else []

    if n == 3:
        # Outlier removal: find the point with distance > ε to both others
        d01 = _dist(cluster[0].world_xy, cluster[1].world_xy)
        d02 = _dist(cluster[0].world_xy, cluster[2].world_xy)
        d12 = _dist(cluster[1].world_xy, cluster[2].world_xy)
        # Which pairs are consistent?
        ok01 = d01 < epsilon_m
        ok02 = d02 < epsilon_m
        ok12 = d12 < epsilon_m
        if ok01 and ok02 and ok12:
            return cluster                      # all three consistent
        if ok01 and not ok02 and not ok12:
            return [cluster[0], cluster[1]]     # P2 is the outlier
        if ok02 and not ok01 and not ok12:
            return [cluster[0], cluster[2]]     # P1 is the outlier
        if ok12 and not ok01 and not ok02:
            return [cluster[1], cluster[2]]     # P0 is the outlier
        return []                               # no consistent pair

    # N > 3: return all (greedy cluster already enforced ε)
    return cluster


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
