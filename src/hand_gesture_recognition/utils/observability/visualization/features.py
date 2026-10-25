from __future__ import annotations

from ._helpers import _feature_matrix, _shared_ylim

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def plot_feature_over_windows(
    processed_dataset: Dataset,
    windowing,
    selection: Dict[str, List[int]],
    features: List[str],
    max_windows: int,
    class_names: List[str],
    run_dir: str,
) -> List[str]:
    """One figure per selected gesture: rows = features, cols = channels,
    value vs window index."""
    paths: List[str] = []
    if processed_dataset.n_trials == 0:
        return paths
    names = (
        class_names
        if class_names is not None
        else processed_dataset.metadata.class_names
    )
    present = set(int(l) for l in np.unique(processed_dataset.labels))
    if getattr(windowing, "mode", "single") == "two_level":
        # Two-level mode filters per big window (small windows are carved from
        # the filtered big window), so features belong to the big windows.
        wd = (
            get_windowing_strategy("two_level")
            .window(processed_dataset, windowing, [])
            .big
        )
    else:
        wd = get_windowing_strategy("recognition").window(
            processed_dataset, windowing, []
        )
    if wd.X.shape[0] == 0:
        return paths
    fs = processed_dataset.metadata.sampling_rate
    X_feat, feat_names = _feature_matrix(wd.X, fs)

    for g_name, channels in selection.items():
        try:
            g_idx = names.index(g_name)
        except ValueError:
            continue
        if g_idx not in present:
            continue
        idx = np.where(processed_dataset.labels == g_idx)[0]
        tid = str(processed_dataset.trial_ids[idx[0]])
        mask = wd.metadata.trial_ids == tid
        if not mask.any():
            continue
        path = os.path.join(run_dir, f"feature_over_windows_{g_name}.png")
        _plot_features(
            X_feat[mask], feat_names, channels, features, max_windows, g_name, path
        )
        paths.append(path)
    return paths


def _plot_features(X_feat, feat_names, channels, features, max_windows, g_name, path):
    wanted = features or list(dict.fromkeys(n.rsplit("_ch", 1)[0] for n in feat_names))
    if max_windows and X_feat.shape[0] > max_windows:
        X_feat = X_feat[:max_windows]
    fig, axes = plt.subplots(
        len(wanted),
        len(channels),
        figsize=(2.5 * len(channels), 1.8 * len(wanted)),
        squeeze=False,
    )
    w = np.arange(X_feat.shape[0])
    for r, feat in enumerate(wanted):
        row_traces = []
        for c_idx, ch in enumerate(channels):
            col = f"{feat}_ch{ch + 1}"
            ax = axes[r, c_idx]
            if col in feat_names:
                j = feat_names.index(col)
                ax.plot(w, X_feat[:, j], lw=0.8)
                row_traces.append(X_feat[:, j])
            else:
                ax.set_visible(False)
            ax.set_xlabel("window")
            if c_idx == 0:
                ax.set_ylabel(feat)
        if row_traces:
            _shared_ylim(axes[r, :], row_traces)
    axes[0, 0].set_title(f"features over windows — {g_name}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
