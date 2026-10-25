from __future__ import annotations

from ._helpers import _channel_axes

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def plot_fft_spectrum(
    dataset: Dataset,
    windowing,
    sampling_rate: Optional[float] = None,
    example_tid: Optional[str] = None,
    window_index: int = 0,
    max_frequency: Optional[float] = None,
    use_hann: bool = True,
    remove_dc: bool = True,
    path: Optional[str] = None,
):
    """FFT power spectrum for one window: one subplot per channel, with the
    extracted peak/mean/median frequency marked. Mirrors the logic in
    ``FrequencyDomainFeatures`` (DC removal + Hann + real FFT)."""
    fs = sampling_rate if sampling_rate is not None else dataset.metadata.sampling_rate
    if getattr(windowing, "mode", "single") == "two_level":
        # Two-level mode filters per big window, so the FFT is over big windows.
        wd = get_windowing_strategy("two_level").window(dataset, windowing, []).big
    else:
        wd = get_windowing_strategy("recognition").window(dataset, windowing, [])
    if wd.X.shape[0] == 0:
        return None
    if window_index >= wd.X.shape[0]:
        window_index = wd.X.shape[0] - 1
    signal = wd.X[window_index].astype(np.float32)  # (n_samples, n_channels)
    n_samples, n_channels = signal.shape

    if remove_dc:
        signal = signal - np.mean(signal, axis=0, keepdims=True)
    if use_hann:
        signal = signal * np.hanning(n_samples).astype(np.float32)[:, None]

    fft = np.fft.rfft(signal, axis=0)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    power = np.abs(fft) ** 2
    power = power[1:, :]
    frequencies = frequencies[1:]

    fig, axes = _channel_axes(n_channels)
    for c in range(n_channels):
        p = power[:, c]
        total = p.sum()
        peak_freq = frequencies[np.argmax(p)]
        weighted = float(np.sum(p * frequencies))
        mean_freq = weighted / total if total > 0 else 0.0
        cumulative = np.cumsum(p)
        median_freq = frequencies[np.argmax(cumulative >= total / 2.0)]

        axes[c].plot(frequencies, p, lw=0.8)
        axes[c].axvline(peak_freq, color="tab:red", lw=1, label="peak")
        axes[c].axvline(mean_freq, color="tab:green", lw=1, label="mean")
        axes[c].axvline(median_freq, color="tab:blue", lw=1, label="median")
        axes[c].set_ylabel(f"ch{c + 1}")
        if max_frequency is not None:
            axes[c].set_xlim(0, max_frequency)

    axes[0].legend(fontsize=7)
    axes[0].set_title(f"FFT power spectrum — window {window_index}")
    axes[-1].set_xlabel("frequency (Hz)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
