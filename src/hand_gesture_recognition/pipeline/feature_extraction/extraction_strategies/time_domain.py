from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    TIME_DOMAIN_FEATURE_NAMES,
)


def _rms(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(X**2, axis=1))


def _std(X: np.ndarray) -> np.ndarray:
    return np.std(X, axis=1)


def _min(X: np.ndarray) -> np.ndarray:
    return np.min(X, axis=1)


def _max(X: np.ndarray) -> np.ndarray:
    return np.max(X, axis=1)


def _mean(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=1)


def _mav(X: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(X), axis=1)


def _iemg(X: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(X), axis=1)


def _variance(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=1)


def _wl(X: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.diff(X, axis=1)), axis=1)


def _max_over_mean(X: np.ndarray) -> np.ndarray:
    mx = np.max(X, axis=1)
    mean = np.mean(X, axis=1)
    return np.divide(mx, mean, out=np.zeros_like(mx, dtype=np.float32), where=mean != 0)


def _min_over_max(X: np.ndarray) -> np.ndarray:
    mn = np.min(X, axis=1)
    mx = np.max(X, axis=1)
    return np.divide(mn, mx, out=np.zeros_like(mn, dtype=np.float32), where=mx != 0)


_COMPUTE = {
    "rms": _rms,
    "std": _std,
    "min": _min,
    "max": _max,
    "mean": _mean,
    "max_over_mean": _max_over_mean,
    "min_over_max": _min_over_max,
    "mav": _mav,
    "iemg": _iemg,
    "variance": _variance,
    "wl": _wl,
}


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel handcrafted time-domain features.

    Computes the features named in ``FEATURE_NAMES`` (or the subset passed via
    ``features``). Each feature is computed per window and per channel, then the
    blocks are concatenated column-wise in feature-major order.
    Output shape: (n_windows, n_selected * n_channels).
    """

    FEATURE_NAMES = TIME_DOMAIN_FEATURE_NAMES

    def __init__(self, features: Optional[List[str]] = None):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_windows, time, channels)
        X = np.asarray(X, dtype=np.float32)
        names = self.features or self.FEATURE_NAMES
        return np.concatenate([_COMPUTE[n](X) for n in names], axis=1)
