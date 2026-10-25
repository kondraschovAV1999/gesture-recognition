from __future__ import annotations

from ._helpers import _shade_gesture, _shared_ylim, _trial_index

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset


def plot_raw_signal(
    dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    channels: Optional[List[int]] = None,
    consistent_scale: bool = False,
):
    """One trial's raw signal for the given channels (0-based), with the
    ground-truth gesture segment shaded. When ``consistent_scale`` is true, all
    channel subplots share one y-range."""
    if dataset.n_trials == 0:
        return None
    tid = example_tid or str(dataset.trial_ids[0])
    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    fs = dataset.metadata.sampling_rate
    n_ch = trial.emg.shape[1]
    if channels is None:
        channels = list(range(n_ch))

    fig, axes = plt.subplots(
        len(channels), 1, figsize=(12, 1.9 * len(channels)), squeeze=False
    )
    axes = axes[:, 0]
    t = np.arange(trial.emg.shape[0]) / fs
    plotted = []
    for row, c in enumerate(channels):
        axes[row].plot(t, trial.emg[:, c], lw=0.6)
        axes[row].set_ylabel(f"ch{c + 1}")
        plotted.append(trial.emg[:, c])
    _shade_gesture(axes, int(dataset.trials[i].start), int(dataset.trials[i].end), fs)
    if consistent_scale:
        _shared_ylim(axes, plotted)

    axes[0].set_title(f"raw signal — {tid}")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_raw_signal_by_class(
    dataset: Dataset,
    run_dir: str,
    class_names: Optional[List[str]] = None,
    selection: Optional[Dict[str, List[int]]] = None,
    consistent_scale: bool = False,
) -> List[str]:
    """Write one raw-signal figure per class. When ``selection`` is provided,
    only its gestures (with their channels) are plotted; otherwise every class
    with all channels (legacy behavior)."""
    # Resolve plot_raw_signal through the package root at call time so the
    # package-level binding (monkeypatchable, as in the pre-split module) is
    # the one this function delegates to.
    from hand_gesture_recognition.utils.observability.visualization import (
        plot_raw_signal,
    )

    paths: List[str] = []
    if dataset.n_trials == 0:
        return paths
    names = class_names if class_names is not None else dataset.metadata.class_names
    if selection is not None:
        present = set(int(l) for l in np.unique(dataset.labels))
        for g_name, channels in selection.items():
            try:
                g_idx = names.index(g_name)
            except ValueError:
                continue
            if g_idx not in present:
                continue
            idx = np.where(dataset.labels == g_idx)[0]
            tid = str(dataset.trial_ids[idx[0]])
            path = os.path.join(run_dir, f"raw_signal_{g_name}.png")
            plot_raw_signal(
                dataset,
                example_tid=tid,
                path=path,
                channels=channels,
                consistent_scale=consistent_scale,
            )
            paths.append(path)
        return paths
    for cls in np.unique(dataset.labels):
        cls = int(cls)
        idx = np.where(dataset.labels == cls)[0]
        tid = str(dataset.trial_ids[idx[0]])
        name = names[cls] if cls < len(names) else str(cls)
        path = os.path.join(run_dir, f"raw_signal_{name}.png")
        plot_raw_signal(dataset, example_tid=tid, path=path)
        paths.append(path)
    return paths
