"""Data augmentation for contact-detection training.

The real (waveshare_work) corpus contains only ~250 contact-positive frames
in total, which is the binding constraint on every contact detector in this
project (see reports/historical_investigations.md §3.2, §4.1, and the
synthetic-data experiment in memory: synthetic frames turned out to sit ~+7.9
sigma from the real distribution and did not substitute for real data). This
module multiplies the effective size of the REAL corpus instead.

Three augmentations, per the project owner's directive:

  * horizontal flip
  * small random rotation, uniform in [-max_rotation_deg, +max_rotation_deg]
  * additive Gaussian "noise poisoning", x ~ N(0, noise_sigma)

Two design decisions that are not arbitrary and are easy to get wrong:

**Geometric transforms are sampled once per RUN, not per frame.** A flip or
rotation models the camera's pose relative to the scene, which is fixed for
the duration of a clip. Sampling a fresh angle per frame would make the scene
appear to jitter, and ``ThermoX3DDetector``'s whole point is a temporal
convolution over T consecutive frames -- it would learn that jitter as
motion. The same transform is likewise applied to all three cameras of a
triplet: they are one rigid multi-view observation of one moment.

**Noise is sampled per frame, i.i.d.** That is what sensor noise actually is
(the Waveshare profile quotes a 0.7 degC noise floor), and it is the one
augmentation here that *should* vary frame-to-frame -- it teaches the
temporal path that per-frame fluctuation is not signal.

**Augment BEFORE preprocessing.** These functions operate on raw thermal
frames in degrees C, so ``noise_sigma`` is interpretable in physical units
(sigma=2 degC is ~3x the sensor's own noise floor, and ~0.8x the real
corpus's frame-wide std of ~2.39 degC). Applying the same sigma after
``GlobalNormPreprocessor`` would be a completely different -- and much more
destructive -- perturbation, because the residual's std is only ~0.9.
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from thermal_algorithms.core.types import Frame

__all__ = ["AugmentParams", "augment_frame_array", "augment_examples"]


class AugmentParams:
    """Container for the augmentation hyperparameters (all Optuna-searchable).

    ``flip_prob`` is the probability that a run is mirrored; ``rotate_prob``
    the probability that it is rotated at all (the angle is then drawn
    uniformly from +/-``max_rotation_deg``); ``noise_sigma`` is in degrees C.
    Setting any to 0 disables that augmentation.
    """

    __slots__ = ("flip_prob", "rotate_prob", "max_rotation_deg", "noise_sigma")

    def __init__(self, *, flip_prob: float = 0.5, rotate_prob: float = 0.5,
                 max_rotation_deg: float = 15.0, noise_sigma: float = 2.0) -> None:
        self.flip_prob = float(flip_prob)
        self.rotate_prob = float(rotate_prob)
        self.max_rotation_deg = float(max_rotation_deg)
        self.noise_sigma = float(noise_sigma)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"AugmentParams(flip_prob={self.flip_prob}, rotate_prob={self.rotate_prob}, "
                f"max_rotation_deg={self.max_rotation_deg}, noise_sigma={self.noise_sigma})")

    def as_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}


def augment_frame_array(
    arr: np.ndarray,
    *,
    flip: bool,
    angle_deg: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one fixed geometric transform (+ fresh noise) to a single (H, W)
    thermal frame. ``flip``/``angle_deg`` are passed in rather than sampled
    here precisely so a caller can hold them constant across a run."""
    out = arr.astype(np.float32, copy=True)
    if flip:
        out = np.ascontiguousarray(out[:, ::-1])
    if angle_deg:
        from scipy.ndimage import rotate as _rotate
        # mode="nearest" replicates the edge pixels. The alternative,
        # constant-0 fill, would paste a 0 degC (i.e. freezing) wedge into the
        # corners of every rotated frame -- an object the sensor can never see
        # and a far larger artifact than the rotation itself.
        out = _rotate(out, angle_deg, axes=(-2, -1), reshape=False,
                      order=1, mode="nearest").astype(np.float32)
    if noise_sigma > 0:
        out = out + rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
    return out


def augment_examples(
    examples: Sequence[tuple],
    params: AugmentParams,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
):
    """Augment one contiguous run of ``(triplet, ContactEvent)`` examples.

    Returns a NEW list; the input is not modified, and the ``ContactEvent``
    labels are passed through unchanged -- none of these transforms can change
    whether contact physically occurred.

    The flip decision and rotation angle are drawn once and shared by every
    frame and every camera in the run (see module docstring); the noise is
    redrawn per frame.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    flip = bool(rng.random() < params.flip_prob)
    angle = 0.0
    if params.max_rotation_deg > 0 and rng.random() < params.rotate_prob:
        angle = float(rng.uniform(-params.max_rotation_deg, params.max_rotation_deg))

    out = []
    for triplet, event in examples:
        new_triplet = tuple(
            replace(f, data=augment_frame_array(
                f.data, flip=flip, angle_deg=angle,
                noise_sigma=params.noise_sigma, rng=rng,
            ))
            for f in triplet
        )
        out.append((new_triplet, event))
    return out


def augment_runs(
    runs: Sequence[Sequence[tuple]],
    params: AugmentParams,
    *,
    n_copies: int = 1,
    seed: int = 0,
):
    """Convenience: produce ``n_copies`` independently-augmented versions of
    each run in ``runs``, returned as a flat list of runs. ``n_copies=0``
    yields nothing; the caller is responsible for also including the
    un-augmented originals if it wants them."""
    rng = np.random.default_rng(seed)
    out = []
    for run in runs:
        for _ in range(n_copies):
            out.append(augment_examples(run, params, rng=rng))
    return out
