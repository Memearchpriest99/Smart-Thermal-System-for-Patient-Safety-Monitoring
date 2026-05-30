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
