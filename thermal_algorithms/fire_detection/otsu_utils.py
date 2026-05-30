"""Shared Otsu segmentation and blob extraction utilities for fire detection.

Both OtsuFireDetector and FireSVMDetector need the same two primitives:
  - otsu_segment()     — variance-gated Otsu thresholding + morphological shaping
  - extract_blobs()    — blob feature extraction from a binary mask

Keeping them here avoids duplication and makes unit-testing the primitives
independent of either detector class.
"""

from __future__ import annotations

import numpy as np
import cv2


def otsu_segment(
    data: np.ndarray,
    morph_kernel: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """Apply Otsu's method with morphological shaping to a float32 thermal frame.

    Args:
        data: 2-D float32 thermal array (already assumed to have sufficient
              dynamic range — variance gating is the caller's responsibility).
        morph_kernel: Structuring element for morphological operations.
        n_bins: Number of histogram bins for Otsu (default 256).

    Returns:
        Binary mask (uint8, 0/255) after 1× erosion + 2× dilation.
    """
    d_min = float(data.min())
    d_max = float(data.max())
    span = d_max - d_min + 1e-6

    # Normalize to integer bin indices for histogram
    norm = ((data - d_min) / span * (n_bins - 1)).astype(np.float32)

    hist, _ = np.histogram(norm.ravel(), bins=n_bins, range=(0.0, n_bins - 1.0))
    hist = hist.astype(np.float64)
    N = hist.sum()
    if N == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    p = hist / N
    idx = np.arange(n_bins, dtype=np.float64)

    # Cumulative class weight ω₀(k) and cumulative mean-numerator μ_k
    w0 = np.cumsum(p)
    mu_k = np.cumsum(idx * p)
    mu_t = mu_k[-1]  # total mean

    # Between-class variance: σ_B²(k) = (μ_t·ω₀ − μ_k)² / (ω₀·ω₁)
    # Derivation: ω₀·ω₁·(μ₁ − μ₀)² = (μ_t·ω₀ − μ_k)² / (ω₀·ω₁)
    w1 = 1.0 - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b2 = np.where(
            (w0 > 0) & (w1 > 0),
            (mu_t * w0 - mu_k) ** 2 / (w0 * w1),
            0.0,
        )

    k_star = int(np.argmax(sigma_b2))
    threshold = d_min + (k_star / (n_bins - 1)) * (d_max - d_min)

    raw_mask = (data >= threshold).astype(np.uint8) * 255

    # §4.4.4 step (f): 1× erosion then 2× dilation
    eroded = cv2.erode(raw_mask, morph_kernel, iterations=1)
    dilated = cv2.dilate(eroded, morph_kernel, iterations=2)
    return dilated


def extract_blobs(data: np.ndarray, mask: np.ndarray) -> list[dict]:
    """Extract per-blob features from a binary ROI mask.

    Args:
        data: Original float32 thermal frame.
        mask: Binary mask (uint8, 0/255) from otsu_segment().

    Returns:
        List of feature dicts, one per connected component (sorted by area desc).
        Each dict has keys: area, max_temp, mean_temp, std_temp,
        centroid_x, centroid_y, skewness, kurtosis.
        Empty list if no blobs found.
    """
    from scipy.stats import skew, kurtosis as kurt

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 1.0:
            continue

        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixels = data[blob_mask > 0].astype(np.float64)
        if pixels.size == 0:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            cx, cy = bx + bw / 2.0, by + bh / 2.0

        blobs.append({
            "area": area,
            "max_temp": float(pixels.max()),
            "mean_temp": float(pixels.mean()),
            "std_temp": float(pixels.std()),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "skewness": float(skew(pixels)),
            "kurtosis": float(kurt(pixels)),
            # Bounding rectangle — used by the Trainer for IoU evaluation (§5.3.4)
            "bbox": (float(bx), float(by), float(bw), float(bh)),
        })

    blobs.sort(key=lambda b: b["area"], reverse=True)
    return blobs
