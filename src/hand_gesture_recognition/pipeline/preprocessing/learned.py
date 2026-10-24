from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PerChannelStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize each channel using mean/std fit on training data only.

    Learned preprocessing: ``fit`` must see train windows only; ``transform``
    is then applied to train/val/test. Input shape ``(..., channels)``; stats
    are computed per channel over the leading axes.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=tuple(range(X.ndim - 1)), keepdims=True)
        self.std_ = X.std(axis=tuple(range(X.ndim - 1)), keepdims=True)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean_) / self.std_
