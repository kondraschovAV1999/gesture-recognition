from __future__ import annotations

import os
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import window_dataset
from hand_gesture_recognition.prediction.base import Prediction


def plot_predictions_on_signal(
    dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    path: Optional[str] = None,
):
    """Plot one example test trial per gesture: channel-1 signal, ground-truth
    gesture segment (green band), and per-window predicted vs expected labels
    (green = correct, red = wrong)."""
    if dataset.n_trials == 0:
        return None
    test_wd = window_dataset(dataset, windowing)
    fs = dataset.metadata.sampling_rate
    ws = test_wd.X.shape[1]

    # pick the first occurrence of each gesture label
    unique_trials = []
    seen_labels = set()
    trial_index = {tid: i for i, tid in enumerate(dataset.trial_ids.tolist())}
    for i in range(dataset.n_trials):
        label = int(dataset.labels[i])
        if label not in seen_labels:
            seen_labels.add(label)
            unique_trials.append(dataset.trial_ids.tolist()[i])

    n = len(unique_trials)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), squeeze=False)
    for row, tid in enumerate(unique_trials):
        ax = axes[row, 0]
        i = trial_index[tid]
        trial = dataset.trials[i]
        g_start, g_end = int(dataset.gesture_start[i]), int(dataset.gesture_end[i])
        true_label = int(dataset.labels[i])
        t = np.arange(trial.shape[0]) / fs

        ax.plot(t, trial[:, 0], color="tab:blue", lw=0.7)
        if g_start >= 0:
            ax.axvspan(g_start / fs, g_end / fs, color="green", alpha=0.22)
        ax.set_title(f"{tid} — expected: {class_names[true_label]}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("ch1")

        mask = test_wd.metadata.trial_ids == tid
        starts = test_wd.metadata.window_start[mask]
        y_pred = prediction.y_pred[mask]
        y_true = test_wd.y[mask]
        ylim = ax.get_ylim()
        for s, yp, yt in zip(starts, y_pred, y_true):
            color = "green" if yp == yt else "red"
            x = (s + ws / 2) / fs
            ax.axvline(s / fs, color=color, alpha=0.25, lw=0.5)
            ax.annotate(
                class_names[int(yp)],
                xy=(x, ylim[1] * 0.9),
                color=color,
                fontsize=6,
                ha="center",
                va="top",
                rotation=90,
            )
        ax.set_ylim(ylim)

    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def plot_signal_data_flow(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    test_dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """4-panel figure: raw signal, signal-processed signal, per-window
    prediction probabilities, and expected-vs-predicted labels."""
    if test_dataset.n_trials == 0:
        return None
    tid = example_tid or str(test_dataset.trial_ids[0])
    fs = processed_dataset.metadata.sampling_rate

    def find_trial(ds, tid):
        for i, t in enumerate(ds.trial_ids.tolist()):
            if t == tid:
                return ds.trials[i]
        return None

    raw_trial = find_trial(raw_dataset, tid)
    proc_trial = find_trial(processed_dataset, tid)
    test_wd = window_dataset(test_dataset, windowing)
    mask = test_wd.metadata.trial_ids == tid

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if raw_trial is not None:
        axes[0, 0].plot(np.arange(raw_trial.shape[0]) / fs, raw_trial, lw=0.6)
    axes[0, 0].set_title("raw signal (8 channels)")
    axes[0, 0].set_xlabel("time (s)")

    if proc_trial is not None:
        axes[0, 1].plot(np.arange(proc_trial.shape[0]) / fs, proc_trial, lw=0.6)
    axes[0, 1].set_title("signal-processed (8 channels)")
    axes[0, 1].set_xlabel("time (s)")

    y_score = prediction.y_score[mask] if prediction.y_score is not None else None
    if y_score is not None and y_score.shape[0] > 0:
        im = axes[1, 0].imshow(y_score.T, aspect="auto", cmap="viridis")
        axes[1, 0].set_yticks(range(len(class_names)))
        axes[1, 0].set_yticklabels(class_names)
        axes[1, 0].set_title("prediction probabilities (window × class)")
        axes[1, 0].set_xlabel("window index")
        fig.colorbar(im, ax=axes[1, 0])

    y_true = test_wd.y[mask]
    y_pred = prediction.y_pred[mask]
    w = np.arange(len(y_true))
    axes[1, 1].plot(w, y_true, "o-", label="expected", lw=1)
    axes[1, 1].plot(w, y_pred, "x--", label="predicted", lw=1)
    axes[1, 1].set_yticks(range(len(class_names)))
    axes[1, 1].set_yticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].set_title("expected vs predicted per window")
    axes[1, 1].set_xlabel("window index")

    fig.suptitle(f"data flow for {tid}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
