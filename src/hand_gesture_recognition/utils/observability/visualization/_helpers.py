from __future__ import annotations

from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import VisualizationSelectionConfig
from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.frequency_domain import (
    FrequencyDomainFeatures,
)
from hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain import (
    TimeDomainFeatures,
)


def _channel_axes(n_channels: int):
    fig, axes = plt.subplots(
        n_channels, 1, figsize=(12, 1.9 * n_channels), squeeze=False
    )
    return fig, axes[:, 0]


def _trial_index(dataset: Dataset, tid: str) -> int:
    for i, t in enumerate(dataset.trial_ids.tolist()):
        if str(t) == tid:
            return i
    raise KeyError(tid)


def _shade_gesture(axes, g_start: int, g_end: int, fs: float) -> None:
    if g_start >= 0:
        for ax in axes:
            ax.axvspan(g_start / fs, g_end / fs, color="green", alpha=0.2)


def _shared_ylim(axes, traces, pad=0.05):
    """Set a single shared y-range across all axes from the given traces."""
    lo, hi = np.inf, -np.inf
    for arr in traces:
        if arr is None or arr.size == 0:
            continue
        lo = min(lo, float(np.nanmin(arr)))
        hi = max(hi, float(np.nanmax(arr)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    span = hi - lo
    for ax in axes:
        ax.set_ylim(lo - pad * span, hi + pad * span)


def resolve_selection(
    selection: VisualizationSelectionConfig, class_names: List[str], n_channels: int
) -> Dict[str, List[int]]:
    """Gesture name -> 0-based channel indices. Empty lists mean 'all'."""
    gestures = selection.gestures or list(class_names)
    all_channels = list(range(n_channels))
    out: Dict[str, List[int]] = {}
    for g in gestures:
        ch = selection.by_gesture.get(g)
        if ch is None:
            ch = selection.channels
        idx = [int(c) - 1 for c in ch] if ch else all_channels
        out[g] = sorted(set(idx))
    return out


def _feature_matrix(X: np.ndarray, sampling_rate: float):
    """(n_windows, 15*n_channels) matrix (11 time-domain + 4 frequency-domain) + flattened per-column names."""
    td = TimeDomainFeatures().transform(X)
    fd = FrequencyDomainFeatures(sampling_rate=sampling_rate).transform(X)
    n_ch = X.shape[-1]
    td_names = [
        f"{n}_ch{c + 1}" for n in TimeDomainFeatures.FEATURE_NAMES for c in range(n_ch)
    ]
    fd_names = [
        f"{n}_ch{c + 1}"
        for n in FrequencyDomainFeatures.FEATURE_NAMES
        for c in range(n_ch)
    ]
    return np.concatenate([td, fd], axis=1), td_names + fd_names
