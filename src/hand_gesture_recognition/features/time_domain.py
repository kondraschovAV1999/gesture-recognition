from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    """Handcrafted time-domain features: RMS, mean absolute value, waveform length."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X: (n_windows, time, channels)
        X = np.asarray(X, dtype=np.float32)
        rms = np.sqrt(np.mean(X**2, axis=1))
        mav = np.mean(np.abs(X), axis=1)
        wl = np.sum(np.abs(np.diff(X, axis=1)), axis=1)
        return np.concatenate([rms, mav, wl], axis=1)
