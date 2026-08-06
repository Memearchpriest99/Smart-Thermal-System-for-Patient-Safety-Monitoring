"""FireSVMDetector — feature-based SVM fire detector (§ 4.4.4, ML approach).

Binary classification: SAFE (−1) vs ACTIVE_COMBUSTION (+1).

Feature vector (§ 4.4.4 + EDA §5.2):
    max_temp, mean_temp, std_temp, area, skewness, kurtosis

Pipeline (Algorithm Flow in §4.4.4):
    1. Region Proposal     — Otsu segmentation to isolate candidate blobs.
    2. Feature Extraction  — Compute the 6 scalar features over the hottest blob.
    3. Standardization     — z-score normalise with μ, σ learned at fit time.
    4. Classification      — SVM decision function: ŷ = sign(w^T φ(x) + b).
    5. Decision Logic      — ŷ = +1 → ACTIVE_COMBUSTION; ŷ = −1 → SAFE.

The kernel (linear or RBF) is selected at construction time and fixed; the
report defers final kernel selection to the training phase.

Resolution behavior: 'invariant' — features are scalar statistics, so the
same trained checkpoint works for both MLX90640 and Waveshare.
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from thermal_algorithms.core.sensor_profile import SensorProfile
from thermal_algorithms.core.types import FireAlert, FireLevel, Frame
from thermal_algorithms.fire_detection.base import FireDetector
from thermal_algorithms.fire_detection.otsu_utils import extract_blobs, otsu_segment


# Feature names, in the order they're assembled into the feature vector.
_FEATURES = ("max_temp", "mean_temp", "std_temp", "area", "skewness", "kurtosis")


def _frame_to_feature_vector(
    data: np.ndarray,
    morph_kernel: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Extract the 6-element feature vector from a single thermal frame."""
    mask = otsu_segment(data, morph_kernel, n_bins)
    blobs = extract_blobs(data, mask)

    if not blobs:
        return np.zeros(len(_FEATURES), dtype=np.float64)

    # Use the hottest blob as the candidate region of interest
    hottest = max(blobs, key=lambda b: b["max_temp"])
    fv = np.array([hottest[k] for k in _FEATURES], dtype=np.float64)

    # skewness/kurtosis are NaN for a zero-variance blob (see
    # otsu_utils._skew_kurtosis — deliberately scipy-compatible there).
    # A perfectly flat blob happens on real data (e.g. a frozen/degenerate
    # sensor frame, or a preprocessor whose background estimate is a single
    # global scalar rather than a spatially-varying one) — treat "no
    # measurable shape" as neutral (0.0) rather than letting NaN reach the
    # scaler/SVM, which raises on non-finite input.
    return np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)


class FireSVMDetector(FireDetector):
    """Feature-based SVM binary fire detector (§ 4.4.4 ML approach)."""

    name = "fire_svm_detector"
    is_trainable = True
    resolution_behavior = "invariant"

    def __init__(
        self,
        sensor_profile: Optional[SensorProfile] = None,
        *,
        kernel: str = "rbf",
        svm_C: float = 1.0,
        gamma: str = "scale",
        morph_kernel_size: int = 3,
        n_bins: int = 256,
        random_state: int = 0,
        class_weight: Optional[str | dict] = None,
    ) -> None:
        """
        Args:
            sensor_profile: Optional. Accepted for uniform construction but not
                required — this detector is resolution-invariant.
            kernel: SVM kernel type. 'rbf' (default) handles non-linear class
                boundaries; 'linear' is faster and interpretable. Final selection
                per §4.4.4 deferred to training phase.
            svm_C: Regularisation parameter C. Larger values → less slack.
            gamma: Kernel coefficient for 'rbf'. 'scale' = 1 / (n_features · Var(X)).
            morph_kernel_size: Structuring element size for Otsu segmentation
                that precedes feature extraction. Must be positive odd.
            n_bins: Histogram bins for Otsu. Default 256.
            random_state: RNG seed for reproducible SVM training.
            class_weight: Passed straight through to sklearn's ``SVC`` —
                ``'balanced'`` reweights inversely proportional to class
                frequency (fire is a minority class in the natural-ratio
                dataset), or an explicit ``{0: w0, 1: w1}`` dict. ``None``
                (default) keeps sklearn's uniform-weight behavior. This is a
                constructor parameter, not a `fit()`-time one, because
                sklearn's `class_weight` is fixed at estimator construction.
        """
        if morph_kernel_size < 1 or morph_kernel_size % 2 == 0:
            raise ValueError(
                f"morph_kernel_size must be a positive odd integer; got {morph_kernel_size}."
            )

        super().__init__(
            sensor_profile=sensor_profile,
            kernel=kernel,
            svm_C=svm_C,
            gamma=gamma,
            morph_kernel_size=morph_kernel_size,
            n_bins=n_bins,
            random_state=random_state,
            class_weight=class_weight,
        )

        self._kernel = kernel
        self._svm_C = float(svm_C)
        self._gamma = gamma
        self._class_weight = class_weight
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size)
        )
        self._n_bins = int(n_bins)
        self._random_state = int(random_state)

        # Learned state — populated by fit()
        self._svm: Optional[SVC] = None
        self._scaler: Optional[StandardScaler] = None

    # ---- Training -------------------------------------------------------

    def fit(
        self,
        X: Iterable[Frame],
        y: Iterable[FireAlert] | None = None,
    ) -> "FireSVMDetector":
        """Train the SVM on labeled frames.

        Args:
            X: Iterable of thermal frames.
            y: Parallel iterable of FireAlerts. Frames whose level is
               IGNITION_SOURCE or ACTIVE_COMBUSTION are treated as Fire (+1);
               SAFE / POTENTIAL_FIRE are treated as Non-Fire (−1).
        """
        if y is None:
            raise ValueError(
                "FireSVMDetector.fit() requires labeled FireAlerts (y=...). "
                "Pass parallel FireAlert objects with the correct FireLevel."
            )

        feature_rows: list[np.ndarray] = []
        labels: list[int] = []

        for frame, alert in zip(X, y):
            fv = _frame_to_feature_vector(
                frame.data.astype(np.float32), self._morph_kernel, self._n_bins
            )
            feature_rows.append(fv)
            is_fire = alert.level in {FireLevel.IGNITION_SOURCE, FireLevel.ACTIVE_COMBUSTION}
            labels.append(1 if is_fire else 0)

        if not feature_rows:
            raise ValueError("fit() received an empty dataset.")

        X_mat = np.vstack(feature_rows)
        y_vec = np.array(labels, dtype=np.int32)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_mat)

        self._svm = SVC(
            kernel=self._kernel,
            C=self._svm_C,
            gamma=self._gamma,
            probability=True,
            random_state=self._random_state,
            class_weight=self._class_weight,
        )
        self._svm.fit(X_scaled, y_vec)
        self._is_fitted = True
        return self

    # ---- Inference -------------------------------------------------------

    def predict(self, X: Frame) -> FireAlert:
        """Classify one frame. Returns SAFE or ACTIVE_COMBUSTION."""
        if not self._is_fitted or self._svm is None or self._scaler is None:
            raise RuntimeError("FireSVMDetector.predict() called before fit().")

        fv = _frame_to_feature_vector(
            X.data.astype(np.float32), self._morph_kernel, self._n_bins
        )
        fv_scaled = self._scaler.transform(fv.reshape(1, -1))

        pred = int(self._svm.predict(fv_scaled)[0])
        # probability of the positive class (index 1 if classes_ = [0, 1])
        proba = float(self._svm.predict_proba(fv_scaled)[0, list(self._svm.classes_).index(1)])

        level = FireLevel.ACTIVE_COMBUSTION if pred == 1 else FireLevel.SAFE
        return FireAlert(
            level=level,
            timestamp=X.timestamp,
            blob_features=dict(zip(_FEATURES, fv.tolist())),
            confidence=proba,
        )

    # ---- Persistence ----------------------------------------------------

    def _state_dict(self) -> dict:
        if self._svm is None or self._scaler is None:
            return {}
        import pickle
        return {
            "svm_pickle": pickle.dumps(self._svm),
            "scaler_pickle": pickle.dumps(self._scaler),
        }

    def _load_state_dict(self, state: dict) -> None:
        if not state:
            return
        import pickle
        self._svm = pickle.loads(state["svm_pickle"])
        self._scaler = pickle.loads(state["scaler_pickle"])
        self._is_fitted = True
