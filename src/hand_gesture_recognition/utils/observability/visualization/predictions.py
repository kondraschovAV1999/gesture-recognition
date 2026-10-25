from __future__ import annotations

from ._helpers import _channel_axes, _shade_gesture, _trial_index

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import Prediction
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def plot_predictions_on_signal(
    dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
    """One example trial's signal, every channel on its own subplot, with the
    ground-truth gesture segment shaded and per-window predicted labels
    annotated on channel 1 (green = correct, red = wrong)."""
    if dataset.n_trials == 0:
        return None
    if getattr(windowing, "mode", "single") == "two_level":
        test_wd = (
            get_windowing_strategy("two_level").window(dataset, windowing, None).big
        )
    else:
        test_wd = get_windowing_strategy(task).window(dataset, windowing, None)
    fs = dataset.metadata.sampling_rate
    ws = test_wd.X.shape[1]

    tid = example_tid
    if tid is None:
        # pick the first gesture trial as a meaningful example (rest trials
        # have start < 0)
        for i in range(dataset.n_trials):
            if int(dataset.trials[i].start) >= 0:
                tid = str(dataset.trial_ids[i])
                break
        tid = tid or str(dataset.trial_ids[0])

    i = _trial_index(dataset, tid)
    trial = dataset.trials[i]
    g_start, g_end = int(dataset.trials[i].start), int(dataset.trials[i].end)
    true_label = int(dataset.labels[i])
    n_ch = trial.emg.shape[1]

    fig, axes = _channel_axes(n_ch)
    t = np.arange(trial.emg.shape[0]) / fs
    for c in range(n_ch):
        axes[c].plot(t, trial.emg[:, c], lw=0.6)
        axes[c].set_ylabel(f"ch{c + 1}")
    _shade_gesture(axes, g_start, g_end, fs)

    # per-window predicted labels on channel 1
    mask = test_wd.metadata.trial_ids == tid
    starts = test_wd.metadata.window_start[mask]
    y_pred = prediction.y_pred[mask]
    y_true = test_wd.y[mask]
    ylim = axes[0].get_ylim()
    for s, yp, yt in zip(starts, y_pred, y_true):
        color = "green" if yp == yt else "red"
        x = (s + ws / 2) / fs
        axes[0].axvline(s / fs, color=color, alpha=0.3, lw=0.5)
        axes[0].annotate(
            class_names[int(yp)],
            xy=(x, ylim[1] * 0.9),
            color=color,
            fontsize=6,
            ha="center",
            va="top",
            rotation=90,
        )
    axes[0].set_ylim(ylim)

    axes[0].set_title(
        f"{tid} — expected: {class_names[true_label]} (green=correct, red=wrong)"
    )
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
