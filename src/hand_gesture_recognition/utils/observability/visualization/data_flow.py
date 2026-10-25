from __future__ import annotations

from ._helpers import _trial_index

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import Prediction
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)


def plot_signal_data_flow(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    test_dataset: Dataset,
    prediction: Prediction,
    windowing,
    class_names: List[str],
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
    task: str = "recognition",
):
    """4-panel data flow: raw ch1, processed ch1, per-window prediction
    probabilities, and expected-vs-predicted labels."""
    if test_dataset.n_trials == 0:
        return None
    tid = example_tid or str(test_dataset.trial_ids[0])
    fs = processed_dataset.metadata.sampling_rate

    def find_trial(ds, tid):
        try:
            return ds.trials[_trial_index(ds, tid)]
        except KeyError:
            return None

    raw_trial = find_trial(raw_dataset, tid)
    proc_trial = find_trial(processed_dataset, tid)
    if getattr(windowing, "mode", "single") == "two_level":
        test_wd = (
            get_windowing_strategy("two_level")
            .window(test_dataset, windowing, None)
            .big
        )
    else:
        test_wd = get_windowing_strategy(task).window(test_dataset, windowing, None)
    mask = test_wd.metadata.trial_ids == tid

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if raw_trial is not None:
        axes[0, 0].plot(
            np.arange(raw_trial.emg.shape[0]) / fs, raw_trial.emg[:, 0], lw=0.7
        )
    axes[0, 0].set_title("raw signal (ch1)")
    axes[0, 0].set_xlabel("time (s)")

    if proc_trial is not None:
        axes[0, 1].plot(
            np.arange(proc_trial.emg.shape[0]) / fs, proc_trial.emg[:, 0], lw=0.7
        )
    axes[0, 1].set_title("signal-processed (ch1)")
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
