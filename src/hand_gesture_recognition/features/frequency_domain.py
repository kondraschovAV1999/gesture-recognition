from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    """Per-channel frequency-domain features using the real FFT.

    Input
    -----
    X : np.ndarray
        Shape: (n_windows, n_samples, n_channels)

    Output
    ------
    np.ndarray
        Shape: (n_windows, 4 * n_channels)

        For each channel:
            1. Total power
            2. Peak frequency
            3. Mean frequency
            4. Median frequency

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.

    use_hann : bool, default=True
        Apply a Hann window before the FFT to reduce spectral leakage.

    remove_dc : bool, default=True
        Remove the mean of each window before computing the FFT.
    """

    FEATURE_NAMES = [
        "total_power",
        "peak_frequency",
        "mean_frequency",
        "median_frequency",
    ]

    def __init__(
        self,
        sampling_rate: float,
        use_hann: bool = True,
        remove_dc: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.use_hann = use_hann
        self.remove_dc = remove_dc

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

        # ---------------------------------------------------------
        # 1. Remove DC component
        # ---------------------------------------------------------
        if self.remove_dc:
            X = X - np.mean(X, axis=1, keepdims=True)

        # ---------------------------------------------------------
        # 2. Apply Hann window for spectral analysis
        # ---------------------------------------------------------
        if self.use_hann:
            window = np.hanning(n_samples).astype(np.float32)
            X_fft = X * window[None, :, None]
        else:
            X_fft = X

        # ---------------------------------------------------------
        # 3. Real FFT along the time axis
        # ---------------------------------------------------------
        fft = np.fft.rfft(X_fft, axis=1)

        # Corresponding frequency bins
        frequencies = np.fft.rfftfreq(
            n_samples,
            d=1.0 / self.sampling_rate,
        )

        # ---------------------------------------------------------
        # 4. Power spectrum
        # ---------------------------------------------------------
        power = np.abs(fft) ** 2

        # Remove DC bin because the signal was already centered.
        power = power[:, 1:, :]
        frequencies = frequencies[1:]

        # ---------------------------------------------------------
        # 5. Total power
        # ---------------------------------------------------------
        total_power = np.sum(power, axis=1)

        # ---------------------------------------------------------
        # 6. Peak / dominant frequency
        # ---------------------------------------------------------
        peak_idx = np.argmax(power, axis=1)

        peak_frequency = frequencies[peak_idx]

        # ---------------------------------------------------------
        # 7. Mean frequency
        # ---------------------------------------------------------
        weighted_frequency = np.sum(
            power * frequencies[None, :, None],
            axis=1,
        )

        mean_frequency = np.divide(
            weighted_frequency,
            total_power,
            out=np.zeros_like(total_power),
            where=total_power > 0,
        )

        # ---------------------------------------------------------
        # 8. Median frequency
        # ---------------------------------------------------------
        cumulative_power = np.cumsum(power, axis=1)

        half_power = total_power[:, None, :] / 2.0

        median_idx = np.argmax(
            cumulative_power >= half_power,
            axis=1,
        )

        median_frequency = frequencies[median_idx]

        # ---------------------------------------------------------
        # 9. Combine features
        # ---------------------------------------------------------
        return np.concatenate(
            [
                total_power,
                peak_frequency,
                mean_frequency,
                median_frequency,
            ],
            axis=1,
        ).astype(np.float32)
