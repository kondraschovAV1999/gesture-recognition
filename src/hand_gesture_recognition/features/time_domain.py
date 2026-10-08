from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel handcrafted time-domain features.

    For each window and each channel, computes: RMS, standard deviation, min,
    max, mean, max/mean, min/max. Output shape: (n_windows, 7 * n_channels).
    """

    FEATURE_NAMES = [
        "rms",
        "std",
        "min",
        "max",
        "mean",
        "max_over_mean",
        "min_over_max",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_windows, time, channels)
        X = np.asarray(X, dtype=np.float32)
        rms = np.sqrt(np.mean(X**2, axis=1))
        std = np.std(X, axis=1)
        mn = np.min(X, axis=1)
        mx = np.max(X, axis=1)
        mean = np.mean(X, axis=1)
        # Ratios guarded against division by zero (flat/rest windows).
        max_over_mean = np.divide(
            mx, mean, out=np.zeros_like(mx, dtype=np.float32), where=mean != 0
        )
        min_over_max = np.divide(
            mn, mx, out=np.zeros_like(mn, dtype=np.float32), where=mx != 0
        )
        return np.concatenate(
            [rms, std, mn, mx, mean, max_over_mean, min_over_max], axis=1
        )
