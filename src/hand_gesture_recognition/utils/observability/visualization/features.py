from __future__ import annotations

from ._helpers import _feature_matrix, _shared_ylim

import os
from typing import Dict, List, Optional

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
    value vs window index.

    In two-level mode features are computed over the SMALL windows (that is
    what the pipeline does — big windows are only preprocessed, never
    feature-extracted); vertical dashed lines mark the big-window boundaries
    so the small windows' grouping is visible.
    """
    paths: List[str] = []
    if processed_dataset.n_trials == 0:
        return paths
    names = (
        class_names
        if class_names is not None
        else processed_dataset.metadata.class_names
    )
    fs = processed_dataset.metadata.sampling_rate

    big_starts: Optional[np.ndarray] = None
    if getattr(windowing, "mode", "single") == "two_level":
        grouped = get_windowing_strategy("two_level").window(
            processed_dataset, windowing, []
        )
        # Feature extraction happens on small windows; big windows only delimit
        # groups. Flatten all small windows and track per-window trial ids and
        # the flat index at which each big window's small windows begin.
        flat: List[np.ndarray] = []
        trial_ids: List[str] = []
        starts: List[int] = []
        offset = 0
        for smalls, tid in zip(grouped.small_windows, grouped.big.metadata.trial_ids):
            n = smalls.shape[0]
            if n == 0:
                continue
            flat.append(smalls)
            trial_ids.extend([str(tid)] * n)
            starts.append(offset)
            offset += n
        if not flat:
            return paths
        X = np.concatenate(flat, axis=0)
        trial_ids_arr = np.asarray(trial_ids)
        big_starts = np.asarray(starts, dtype=int)
    else:
        wd = get_windowing_strategy("recognition").window(
            processed_dataset, windowing, []
        )
        X = wd.X
        trial_ids_arr = wd.metadata.trial_ids.astype(str)

    if X.shape[0] == 0:
        return paths
    X_feat, feat_names = _feature_matrix(X, fs)

    present = set(int(l) for l in np.unique(processed_dataset.labels))
    for g_name, channels in selection.items():
        try:
            g_idx = names.index(g_name)
        except ValueError:
            continue
        if g_idx not in present:
            continue
        idx = np.where(processed_dataset.labels == g_idx)[0]
        tid = str(processed_dataset.trial_ids[idx[0]])
        mask = trial_ids_arr == tid
        if not mask.any():
            continue

        boundaries = None
        if big_starts is not None:
            mask_idx = np.flatnonzero(mask)
            lo, hi = int(mask_idx[0]), int(mask_idx[-1])
            # Interior big-window starts within this trial's small windows.
            rel = big_starts[(big_starts > lo) & (big_starts <= hi)] - lo
            boundaries = rel.tolist() if rel.size else None

        path = os.path.join(run_dir, f"feature_over_windows_{g_name}.png")
        _plot_features(
            X_feat[mask],
            feat_names,
            channels,
            features,
            max_windows,
            g_name,
            path,
            boundaries=boundaries,
        )
        paths.append(path)
    return paths


def _plot_features(
    X_feat,
    feat_names,
    channels,
    features,
    max_windows,
    g_name,
    path,
    boundaries: Optional[List[int]] = None,
):
    wanted = features or list(dict.fromkeys(n.rsplit("_ch", 1)[0] for n in feat_names))
    if max_windows and X_feat.shape[0] > max_windows:
        X_feat = X_feat[:max_windows]
        if boundaries is not None:
            boundaries = [b for b in boundaries if b < max_windows]
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
                # Mark big-window boundaries (only once, on the first row's
                # first column, but repeat per-axis is harmless).
                for b in boundaries or []:
                    ax.axvline(b - 0.5, color="gray", linestyle="--", lw=0.6, alpha=0.7)
            else:
                ax.set_visible(False)
            ax.set_xlabel("small window")
            if c_idx == 0:
                ax.set_ylabel(feat)
        if row_traces:
            _shared_ylim(axes[r, :], row_traces)
    axes[0, 0].set_title(f"features over small windows — {g_name}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
