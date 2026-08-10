"""RBTCTDetector — Rule-Based Temporal Contact Tracker (§ 4.4.3.4).

Promoted from the evaluation-time "config-D" pipeline that lived in
``scripts/`` (``eval_waveshare_contact_v9.py`` + ``_v8.py`` + ``_v3_variants.py``
+ ``ablate_preprocessing.py``). The full mathematical derivation is in
``reports/algorithm_derivations.md`` §8.

Motivation
----------
Both homography-based contact detectors in this package
(``GeometricContactDetector``, ``MVSTGCNDetector``) measure inter-person
distance on a floor plane recovered by homography. On this project's data the
homography's accuracy (~±15 px) is nearly as large as the contact decision
threshold itself (δ≈18 px), so the measurement is barely more informative than
noise. ``ThermoX3DDetector`` avoids that by learning from pixels, but the
corpus contains only ~246 contact-positive frames across 5 scenes, which is
too few to train a 3-D CNN that generalises across scenarios.

RBTCT sidesteps both problems. It observes that **when two people touch, their
thermal signatures merge into a single connected warm region while the person
detector still reports two boxes** — that disagreement is the contact signal,
it is measured entirely in image space (no homography), and it requires **no
contact-labelled training data at all** (only a handful of thresholds, and a
temporal gate that can be calibrated on a small labelled sequence).

Pipeline (per camera, per frame)
--------------------------------
1. Merge over-segmented person boxes (one person split into two detections
   would otherwise fake the "two boxes" half of the signal).
2. Threshold the thermal residual adaptively at ``μ + k·σ``, connected-
   component label it, and count how many box centres fall in each component.
3. Emit a per-camera state:

   ==========  ===========================================================
   ``C``       clear      — no boxes, or two boxes well separated
   ``M``       ambiguous  — exactly one box (which may BE two merged people)
   ``T``       touch      — a component contains the required number of centres
   ``N``       near       — two+ boxes, not merged, but closer than τ_near
   ==========  ===========================================================

4. Combine the three cameras with a **veto-based quorum** (§8.4): at least one
   camera must positively see a merge, a second must be consistent with it,
   and *any* camera that cleanly resolves two separated bodies vetoes.

5. Apply a **causal temporal gate** — see the note below.

Causal temporal gating (differs from the offline evaluation!)
-------------------------------------------------------------
The original config-D applied 1-D morphological opening/closing to the whole
per-scene decision stream offline. Closing is **acausal**: filling a gap
requires knowing that a later frame is positive, which a streaming detector
cannot know. Rather than silently ship an acausal rule as if it were
real-time, this class implements the standard causal analogue:

* opening  → ``attack_frames``  : require N consecutive raw positives before
  asserting contact (rejects momentary blips), and
* closing  → ``release_frames`` : hold the alarm asserted for N frames after
  the raw decision drops (bridges dropouts inside a sustained contact).

This is an attack/release gate, is causal, and introduces at most
``attack_frames`` of latency. It is *close to* but **not identical with**
offline morphology, so numbers produced by this class will differ slightly
from the historical offline config-D figures; ``scripts/eval_config_d.py``
measures the offline variant if that comparison is needed.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from thermal_algorithms.contact_detection.base import (
    ContactDetector,
    ThreeViewDetections,
    ThreeViewFrames,
)
from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import ContactEvent, HomographyMatrices

__all__ = ["RBTCTDetector"]


# ---------------------------------------------------------------------------
# Box geometry (ported verbatim in behaviour from the config-D scripts)
# ---------------------------------------------------------------------------

def _inter(a, b) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    return iw * ih


def _iou(a, b) -> float:
    inter = _inter(a, b)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def _box_gap(a, b) -> float:
    """Minimum edge-to-edge distance between two (x, y, w, h) boxes; 0 if they
    overlap."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    dx = max(0.0, max(a[0], b[0]) - min(ax2, bx2))
    dy = max(0.0, max(a[1], b[1]) - min(ay2, by2))
    return math.hypot(dx, dy)


def _min_gap(boxes) -> float:
    if len(boxes) < 2:
        return math.inf
    return min(_box_gap(boxes[i], boxes[j])
               for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


def _merge_oversegmented(boxes, merge_iou: float, merge_contain: float):
    """Greedily merge boxes that are likely one person (high IoU, or one
    largely contained in the other) into their union box, to a fixpoint.

    The containment term does the real work: a small box entirely inside a
    larger one has low IoU — an IoU-only rule would keep both and fake the
    "two people" precondition — but containment 1.0.
    """
    bs = [tuple(b) for b in boxes]
    changed = True
    while changed and len(bs) > 1:
        changed = False
        for i in range(len(bs)):
            for j in range(i + 1, len(bs)):
                a, b = bs[i], bs[j]
                inter = _inter(a, b)
                min_area = min(a[2] * a[3], b[2] * b[3]) or 1.0
                if _iou(a, b) >= merge_iou or inter / min_area >= merge_contain:
                    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    bs = [bs[k] for k in range(len(bs)) if k not in (i, j)]
                    bs.append((x1, y1, x2 - x1, y2 - y1))
                    changed = True
                    break
            if changed:
                break
    return bs


class RBTCTDetector(ContactDetector):
    """Rule-Based Temporal Contact Tracker — homography-free, training-free."""

    name = "rbtct_detector"
    is_trainable = False
    resolution_behavior = "parameterized"

    def __init__(
        self,
        sensor_profile: SensorProfile,
        *,
        homography: Optional[HomographyMatrices] = None,
        k: float = 1.0,
        tau_near_px: float = 8.0,
        merge_iou: float = 0.2,
        merge_contain: float = 0.7,
        attack_frames: int = 1,
        release_frames: int = 7,
        exact_two_in_crowd: bool = True,
    ) -> None:
        """
        homography: accepted for interface compatibility with the other
            ContactDetectors and STORED, but never read — RBTCT is
            homography-free by construction, which is the entire point of it.
        k: residual threshold is ``mean + k*std`` per frame per camera.
        tau_near_px: below this minimum edge-to-edge box gap, a non-merged
            two-body frame is "near" rather than "clear" (so it does not veto).
        attack_frames: consecutive raw-positive frames required to assert
            contact (causal opening). 1 = assert immediately.
        release_frames: frames to hold the assertion after the raw decision
            drops (causal closing).
        exact_two_in_crowd: with 3+ people visible, require a warm component
            containing EXACTLY two box centres. A component absorbing three or
            more is far more likely a group standing close, or a segmentation
            failure, than a pairwise contact event.
        """
        if sensor_profile is None:
            raise ValueError("RBTCTDetector requires a SensorProfile.")
        super().__init__(
            sensor_profile=sensor_profile,
            homography=homography,
            k=k,
            tau_near_px=tau_near_px,
            merge_iou=merge_iou,
            merge_contain=merge_contain,
            attack_frames=attack_frames,
            release_frames=release_frames,
            exact_two_in_crowd=exact_two_in_crowd,
        )
        self._k = float(k)
        self._tau_near = float(tau_near_px)
        self._merge_iou = float(merge_iou)
        self._merge_contain = float(merge_contain)
        self._attack = int(attack_frames)
        self._release = int(release_frames)
        self._exact_two = bool(exact_two_in_crowd)

        self._run_pos = 0        # consecutive raw positives
        self._hold = 0           # remaining release-hold frames
        self._is_fitted = True   # rule-based: usable out of the box

    # ---- Core rule ---------------------------------------------------------

    def _component_counts(self, resid: np.ndarray, boxes) -> dict:
        from scipy.ndimage import label as cc_label
        thr = float(resid.mean() + self._k * resid.std())
        lab, _n = cc_label(resid > thr)
        h, w = resid.shape
        counts: dict = {}
        for b in boxes:
            cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
            ll = lab[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))]
            if ll > 0:
                counts[ll] = counts.get(ll, 0) + 1
        return counts

    def _camera_state(self, boxes, resid: np.ndarray) -> str:
        n = len(boxes)
        if n == 0:
            return "C"
        if n == 1:
            # NOT negative: a single box may itself be two merged people,
            # which is the exact failure mode this detector exists to survive.
            return "M"
        counts = self._component_counts(resid, boxes)
        any_merge = any(v >= 2 for v in counts.values())
        pair_merge = any(v == 2 for v in counts.values())
        touch = any_merge if (n == 2 or not self._exact_two) else pair_merge
        if touch:
            return "T"
        return "N" if _min_gap(boxes) < self._tau_near else "C"

    def raw_decision(self, X: ThreeViewFrames,
                     detections: Optional[ThreeViewDetections]) -> int:
        """Stateless per-frame decision, BEFORE the temporal gate.

        Exposed publicly because calibration wants to compute these once for a
        whole recording (the expensive part) and then grid-search the cheap
        temporal parameters over the cached streams.
        """
        if detections is None:
            return 0
        boxes = [
            _merge_oversegmented([d.bbox for d in (dets or [])],
                                 self._merge_iou, self._merge_contain)
            for dets in detections
        ]
        if max((len(b) for b in boxes), default=0) <= 1:
            return 0
        states = [self._camera_state(boxes[c], X[c].data.astype(np.float32))
                  for c in range(3)]
        votes = states.count("T")
        merged = states.count("M")
        clear = "C" in states
        return 1 if (votes >= 1 and votes + merged >= 2 and not clear) else 0

    def _gate(self, raw: int) -> int:
        """Causal attack/release gate (see module docstring)."""
        if raw:
            self._run_pos += 1
        else:
            self._run_pos = 0
        if self._run_pos >= max(1, self._attack):
            self._hold = self._release
            return 1
        if self._hold > 0:
            self._hold -= 1
            return 1
        return 0

    # ---- ThermalAlgorithm API ---------------------------------------------

    def fit(self, X: Iterable = (), y: Iterable | None = None) -> "RBTCTDetector":
        """No-op — RBTCT learns nothing from data.

        Exactly the same contract as ``OtsuFireDetector`` and
        ``AdaptiveThresholdDetector``: the tunable parts are thresholds, and
        they are set by calibration (``calibrate_temporal``) rather than by
        gradient descent or an sklearn estimator.
        """
        self._is_fitted = True
        return self

    def predict(self, X: ThreeViewFrames,
                detections: Optional[ThreeViewDetections] = None) -> ContactEvent:
        """One timestep. ``X`` must be the RESIDUAL (background-subtracted)
        frames — the adaptive ``μ + kσ`` threshold is meaningless on raw
        temperature, where σ is dominated by static scene structure rather
        than by people (derived in algorithm_derivations.md §8.6).
        ``detections`` are per-camera person boxes."""
        timestamp = max(f.timestamp for f in X)
        raw = self.raw_decision(X, detections)
        alerted = self._gate(raw)
        return ContactEvent(
            actors=(),
            pairs_in_contact=((0, 1),) if alerted else (),
            timestamp=timestamp,
            confidence=float(alerted),
            debug={"raw": raw, "run_pos": self._run_pos, "hold": self._hold},
        )

    def reset(self) -> None:
        self._run_pos = 0
        self._hold = 0

    # ---- Calibration -------------------------------------------------------

    def calibrate_temporal(
        self,
        streams: Sequence[tuple[Sequence[int], Sequence[int]]],
        *,
        attack_grid: Sequence[int] = (1, 2, 3, 4, 5),
        release_grid: Sequence[int] = tuple(range(0, 9)),
    ) -> tuple[int, int]:
        """Grid-search ``(attack_frames, release_frames)`` to maximise F1 over
        ``streams`` — a sequence of ``(raw_decisions, labels)`` pairs, one per
        recording (never concatenate recordings: the gate carries state, and
        running it across a clip boundary would bleed one scene's hold into
        the next). Sets and returns the best pair."""
        from thermal_algorithms.training.metrics import binary_confusion_matrix

        best = None
        for a in attack_grid:
            for r in release_grid:
                yt, yp = [], []
                for raw, labels in streams:
                    self._attack, self._release = a, r
                    self.reset()
                    yp.extend(self._gate(int(v)) for v in raw)
                    yt.extend(int(v) for v in labels)
                if not yt:
                    continue
                cm = binary_confusion_matrix(yt, yp)
                if best is None or (cm.f1, cm.recall) > (best[2], best[3]):
                    best = (a, r, cm.f1, cm.recall)
        if best is None:
            raise ValueError("calibrate_temporal() received no usable streams.")
        self._attack, self._release = best[0], best[1]
        self.set_params(attack_frames=best[0], release_frames=best[1])
        self.reset()
        return best[0], best[1]

    # ---- Persistence -------------------------------------------------------

    def _state_dict(self) -> dict:
        return {"attack_frames": self._attack, "release_frames": self._release,
                "k": self._k, "tau_near_px": self._tau_near,
                "merge_iou": self._merge_iou, "merge_contain": self._merge_contain,
                "exact_two_in_crowd": self._exact_two}

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        self._attack = int(state.get("attack_frames", self._attack))
        self._release = int(state.get("release_frames", self._release))
        self._k = float(state.get("k", self._k))
        self._tau_near = float(state.get("tau_near_px", self._tau_near))
        self._merge_iou = float(state.get("merge_iou", self._merge_iou))
        self._merge_contain = float(state.get("merge_contain", self._merge_contain))
        self._exact_two = bool(state.get("exact_two_in_crowd", self._exact_two))
        self._is_fitted = True
