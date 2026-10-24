from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d
from sklearn.base import BaseEstimator, TransformerMixin


class RectifyTransformer(BaseEstimator, TransformerMixin):
    """Full-wave rectification: |x| (pointwise, no learning)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.abs(np.asarray(X, dtype=np.float32))


class ButterworthFilter(BaseEstimator, TransformerMixin):
    """Zero-phase Butterworth filter along the time axis (axis=-2).

    Deterministic; no learning. Types:

    - ``bandpass``: ``lowcut``/``highcut`` (Hz)
    - ``bandstop``: ``lowcut``/``highcut`` (Hz)
    - ``lowpass``:  ``highcut`` (Hz)
    - ``highpass``: ``lowcut`` (Hz)
    - ``notch``:    ``frequency`` (Hz)
    """

    def __init__(
        self,
        type: str = "bandpass",
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        frequency: Optional[float] = None,
        order: int = 4,
        fs: float = 200.0,
    ):
        self.type = type
        self.lowcut = lowcut
        self.highcut = highcut
        self.frequency = frequency
        self.order = order
        self.fs = fs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        b, a = self._design()
        return scipy_signal.filtfilt(b, a, X, axis=-2)

    def _design(self):
        nyq = 0.5 * self.fs
        if self.type == "bandpass":
            return scipy_signal.butter(
                self.order, [self.lowcut / nyq, self.highcut / nyq], btype="bandpass"
            )
        if self.type == "bandstop":
            return scipy_signal.butter(
                self.order, [self.lowcut / nyq, self.highcut / nyq], btype="bandstop"
            )
        if self.type == "lowpass":
            return scipy_signal.butter(self.order, self.highcut / nyq, btype="lowpass")
        if self.type == "highpass":
            return scipy_signal.butter(self.order, self.lowcut / nyq, btype="highpass")
        if self.type == "notch":
            return scipy_signal.iirnotch(self.frequency, Q=30.0, fs=self.fs)
        raise ValueError(f"unknown filter type: {self.type!r}")


class EnvelopeExtractor(BaseEstimator, TransformerMixin):
    """Moving-average envelope of the rectified signal (deterministic)."""

    def __init__(self, window_size: int = 25):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.abs(np.asarray(X, dtype=np.float32))
        return uniform_filter1d(X, size=self.window_size, axis=-2, mode="nearest")
