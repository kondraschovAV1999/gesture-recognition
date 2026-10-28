from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from hand_gesture_recognition.configuration.schema.feature_extraction import (
    FREQUENCY_DOMAIN_FEATURE_NAMES,
)


def _total_power(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    return np.sum(power, axis=1)


def _peak_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    peak_idx = np.argmax(power, axis=1)
    return frequencies[peak_idx]


def _mean_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    total = np.sum(power, axis=1)
    weighted = np.sum(power * frequencies[None, :, None], axis=1)
    return np.divide(weighted, total, out=np.zeros_like(total), where=total > 0)


def _median_frequency(power: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    total = np.sum(power, axis=1)
    cumulative = np.cumsum(power, axis=1)
    half_power = total[:, None, :] / 2.0
    median_idx = np.argmax(cumulative >= half_power, axis=1)
    return frequencies[median_idx]


_COMPUTE = {
    "total_power": _total_power,
    "peak_frequency": _peak_frequency,
    "mean_frequency": _mean_frequency,
    "median_frequency": _median_frequency,
}


class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel frequency-domain features using the real FFT.

    Input
    -----
    X : np.ndarray
        Shape: (n_windows, n_samples, n_channels)

    Output
    ------
    np.ndarray
        Shape: (n_windows, n_selected * n_channels)

        For each selected channel-feature pair: total power, peak frequency,
        mean frequency, and/or median frequency.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.

    use_hann : bool, default=True
        Apply a Hann window before the FFT to reduce spectral leakage.

    remove_dc : bool, default=True
        Remove the mean of each window before computing the FFT.

    features : list of str, optional
        Subset of ``FEATURE_NAMES`` to compute. ``None`` (default) computes all.
        ``mean_frequency`` and ``median_frequency`` compute ``total_power``
        internally because they depend on it, but it is only emitted when
        requested.
    """

    FEATURE_NAMES = FREQUENCY_DOMAIN_FEATURE_NAMES

    def __init__(
        self,
        sampling_rate: float,
        use_hann: bool = True,
        remove_dc: bool = True,
        features: Optional[List[str]] = None,
    ):
        self.sampling_rate = sampling_rate
        self.use_hann = use_hann
        self.remove_dc = remove_dc
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract frequency-domain features from windowed sEMG."""
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(
                "Expected X with shape "
                "(n_windows, n_samples, n_channels), "
                f"got shape {X.shape}"
            )

        n_windows, n_samples, n_channels = X.shape

        if n_samples < 2:
            raise ValueError("Each window must contain at least 2 samples.")

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be greater than 0.")

        # 1. Remove DC component
        if self.remove_dc:
            X = X - np.mean(X, axis=1, keepdims=True)

        # 2. Apply Hann window for spectral analysis
        if self.use_hann:
            window = np.hanning(n_samples).astype(np.float32)
            X_fft = X * window[None, :, None]
        else:
            X_fft = X

        # 3. Real FFT along the time axis
        fft = np.fft.rfft(X_fft, axis=1)
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / self.sampling_rate)

        # 4. Power spectrum (drop the DC bin)
        power = np.abs(fft) ** 2
        power = power[:, 1:, :]
        frequencies = frequencies[1:]

        names = self.features if self.features is not None else self.FEATURE_NAMES
        return np.concatenate(
            [_COMPUTE[n](power, frequencies) for n in names], axis=1
        ).astype(np.float32)
