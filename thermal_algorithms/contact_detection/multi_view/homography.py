"""Homography calibration and foot-point projection (§ 4.4.3).

Hot-Point Calibration
---------------------
Standard checkerboard calibration fails in thermal because printed paper has
uniform emissivity.  Instead we use heated calibration targets at known floor
positions ("Hot Point Calibration").

For each camera k, a set of N ≥ 4 point correspondences
    { (u_i, v_i) ↔ (X_w_i, Y_w_i) }
establishes the planar homography H_k via the DLT (Direct Linear Transform):

    [wX_w, wY_w, w]^T  ≅  H_k · [u, v, 1]^T

Each correspondence yields two linear equations in the 9 unknowns h = vec(H_k):

    row_1: [-u, -v, -1,  0,  0,  0,  u·X_w, v·X_w, X_w] · h = 0
    row_2: [ 0,  0,  0, -u, -v, -1,  u·Y_w, v·Y_w, Y_w] · h = 0

Stacking all 2N rows gives A h = 0.  The solution is the right singular
vector of A corresponding to its smallest singular value, reshaped to 3×3
and normalised so that H[2,2] = 1.

Projection
----------
Once calibrated, the world foot-point of a detection is:

    p_world = H_k · [u, v, 1]^T   (homogeneous division gives X_w, Y_w)
"""

from __future__ import annotations

import math

import numpy as np

from thermal_algorithms.core.types import HomographyMatrices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_homography_from_markers(
    marker_correspondences: list[
        tuple[int, list[tuple[tuple[float, float], tuple[float, float]]]]
    ],
) -> HomographyMatrices:
    """Solve for all three camera homography matrices via DLT + SVD.

    Args:
        marker_correspondences: A list of (camera_id, point_pairs) tuples,
            one per camera.  ``point_pairs`` is a list of
            ``((u, v), (X_w, Y_w))`` — image pixel to floor-plane meters.
            Each camera must have at least 4 pairs; more improves accuracy.

    Returns:
        ``HomographyMatrices`` with h1, h2, h3 filled in.  Cameras not
        present in ``marker_correspondences`` get an identity-like fallback
        that passes validation but projects all pixels to (0, 0).

    Raises:
        ValueError: If any camera has fewer than 4 point pairs.
    """
    solved: dict[int, np.ndarray] = {}

    for camera_id, pairs in marker_correspondences:
        if len(pairs) < 4:
            raise ValueError(
                f"Camera {camera_id} has {len(pairs)} point pair(s); "
                f"at least 4 are required for a unique DLT solution."
            )
        solved[camera_id] = _dlt_solve(pairs)

    fallback = np.eye(3, dtype=np.float64)
    h1 = solved.get(0, fallback).copy()
    h2 = solved.get(1, fallback).copy()
    h3 = solved.get(2, fallback).copy()
    return HomographyMatrices(h1=h1, h2=h2, h3=h3)


def project_foot_point(
    uv: tuple[float, float],
    H: np.ndarray,
) -> tuple[float, float]:
    """Project an image foot-point (u, v) to world coordinates (X_w, Y_w).

    Args:
        uv: Pixel coordinates of the bottom-centre of a bounding box.
        H: 3×3 homography matrix for that camera.

    Returns:
        (X_w, Y_w) in metres on the floor plane.
    """
    u, v = float(uv[0]), float(uv[1])
    p_img = np.array([u, v, 1.0], dtype=np.float64)
    p_world = H @ p_img
    w = p_world[2]
    if abs(w) < 1e-9:
        return (0.0, 0.0)
    return (float(p_world[0] / w), float(p_world[1] / w))


# ---------------------------------------------------------------------------
# Internal — DLT solver
# ---------------------------------------------------------------------------

def _dlt_solve(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> np.ndarray:
    """Solve for a single 3×3 homography via Direct Linear Transform.

    Builds the 2N×9 constraint matrix A and returns the right singular
    vector of A corresponding to the smallest singular value, reshaped and
    normalised so H[2, 2] = 1.
    """
    rows: list[np.ndarray] = []
    for (u, v), (xw, yw) in pairs:
        u, v, xw, yw = float(u), float(v), float(xw), float(yw)
        rows.append(np.array([-u, -v, -1.,  0.,  0.,  0.,  u*xw, v*xw, xw]))
        rows.append(np.array([ 0.,  0.,  0., -u, -v, -1.,  u*yw, v*yw, yw]))

    A = np.stack(rows, axis=0)               # (2N, 9)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]                               # smallest right singular vector
    H = h.reshape(3, 3)

    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H.astype(np.float64)


# ---------------------------------------------------------------------------
# Self-calibration from a shared point track (no world coordinates needed)
# ---------------------------------------------------------------------------
#
# When the *same* subject is visible in all three cameras simultaneously, the
# subject's foot-point in each view is an image of one common floor point.
# Across many frames this yields, for any camera pair (k, ref), a set of
# image-to-image correspondences that are all induced by the floor plane —
# exactly the input for a planar homography H_{k->ref}.
#
# Choosing one camera (``ref_camera``) as the reference, we get:
#     H_ref = I              (reference view is its own "world")
#     H_k   = H_{k->ref}     (reprojects view k onto the reference floor plane)
#
# These slot directly into ``HomographyMatrices`` and ``fuse_detections`` with
# no change to the fusion logic. The "world" units are reference-camera floor
# pixels (perspective-distorted, *not* metres) — clustering / contact
# thresholds must be tuned empirically in that plane.


def solve_homography_ransac(
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    threshold_px: float = 2.5,
    iters: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Robustly fit a planar homography src->dst with RANSAC + DLT refit.

    Args:
        pairs: ``((u_src, v_src), (u_dst, v_dst))`` correspondences. Both sides
            are pixels here (image-to-image), but the solver is identical to
            the pixel-to-world case.
        threshold_px: Inlier reprojection-error threshold in destination pixels.
        iters: Number of random 4-point minimal samples to try.
        seed: RNG seed for reproducible sampling.

    Returns:
        ``(H, inlier_indices, inlier_residuals)`` where ``H`` is refit on all
        inliers and residuals are the reprojection errors of those inliers.

    Raises:
        ValueError: If fewer than 4 correspondences are supplied.
    """
    n = len(pairs)
    if n < 4:
        raise ValueError(f"Need >= 4 correspondences for a homography; got {n}.")

    rng = np.random.default_rng(seed)
    best_inliers: list[int] = []

    def residuals_for(H: np.ndarray) -> list[float]:
        out = []
        for (s, d) in pairs:
            px, py = project_foot_point(s, H)
            out.append(math.hypot(px - d[0], py - d[1]))
        return out

    for _ in range(iters):
        sample_idx = rng.choice(n, size=4, replace=False)
        sample = [pairs[i] for i in sample_idx]
        try:
            H = _dlt_solve(sample)
        except np.linalg.LinAlgError:
            continue
        res = residuals_for(H)
        inliers = [i for i, e in enumerate(res) if e < threshold_px]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 4:
        # Degenerate / no consensus — fall back to a plain least-squares fit.
        best_inliers = list(range(n))

    H = _dlt_solve([pairs[i] for i in best_inliers])
    res = residuals_for(H)
    inlier_residuals = [res[i] for i in best_inliers]
    return H, best_inliers, inlier_residuals


def calibrate_floor_homographies_from_tracks(
    points_per_frame: list[dict[int, tuple[float, float]]],
    *,
    ref_camera: int = 0,
    threshold_px: float = 2.5,
    iters: int = 2000,
    seed: int = 0,
) -> tuple[HomographyMatrices, dict[int, dict]]:
    """Self-calibrate floor homographies from a synchronized single-subject track.

    Args:
        points_per_frame: One dict per frame mapping ``camera_id -> (u, v)``
            foot-point. Cameras where the subject was not detected in that
            frame are simply absent from the dict.
        ref_camera: Which camera defines the reference floor plane (its H is
            identity). The other cameras are mapped onto this plane.
        threshold_px: RANSAC inlier threshold (reference-plane pixels).
        iters: RANSAC iterations per camera pair.
        seed: RNG seed.

    Returns:
        ``(HomographyMatrices, info)`` where ``info[cam] = {n_pairs, n_inliers,
        median_residual_px}`` per non-reference camera. The reference camera's
        matrix is the identity.

    Raises:
        ValueError: If any non-reference camera shares < 4 frames with the
            reference camera.
    """
    Hs: dict[int, np.ndarray] = {ref_camera: np.eye(3, dtype=np.float64)}
    info: dict[int, dict] = {}

    for cam in (0, 1, 2):
        if cam == ref_camera:
            info[cam] = {"n_pairs": 0, "n_inliers": 0, "median_residual_px": 0.0,
                         "reference": True}
            continue
        pairs = [
            (fp[cam], fp[ref_camera])
            for fp in points_per_frame
            if cam in fp and ref_camera in fp
        ]
        if len(pairs) < 4:
            raise ValueError(
                f"Camera {cam} shares only {len(pairs)} frame(s) with reference "
                f"camera {ref_camera}; need >= 4 for self-calibration."
            )
        H, inliers, residuals = solve_homography_ransac(
            pairs, threshold_px=threshold_px, iters=iters, seed=seed,
        )
        Hs[cam] = H
        info[cam] = {
            "n_pairs": len(pairs),
            "n_inliers": len(inliers),
            "median_residual_px": float(np.median(residuals)) if residuals else 0.0,
            "reference": False,
        }

    return (
        HomographyMatrices(h1=Hs[0], h2=Hs[1], h3=Hs[2]),
        info,
    )
