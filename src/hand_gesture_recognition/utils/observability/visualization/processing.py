from __future__ import annotations

from ._helpers import _trial_index

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hand_gesture_recognition.configuration.schema import Dataset


def plot_signal_processing_comparison(
    raw_dataset: Dataset,
    processed_dataset: Dataset,
    example_tid: Optional[str] = None,
    path: Optional[str] = None,
):
    """Before/after comparison of signal processing: raw vs processed signal,
    one row per channel (2 columns), for a representative non-noGesture trial."""
    if raw_dataset.n_trials == 0:
        return None
    tid = example_tid
    if tid is None:
        for i in range(raw_dataset.n_trials):
            if int(raw_dataset.trials[i].start) >= 0:
                tid = str(raw_dataset.trial_ids[i])
                break
        tid = tid or str(raw_dataset.trial_ids[0])

    i = _trial_index(raw_dataset, tid)
    raw_trial = raw_dataset.trials[i]
    proc_trial = processed_dataset.trials[_trial_index(processed_dataset, tid)]
    fs = raw_dataset.metadata.sampling_rate
    n_ch = raw_trial.emg.shape[1]

    fig, axes = plt.subplots(n_ch, 2, figsize=(12, 1.8 * n_ch), squeeze=False)
    t_raw = np.arange(raw_trial.emg.shape[0]) / fs
    t_proc = np.arange(proc_trial.emg.shape[0]) / fs
    for c in range(n_ch):
        axes[c, 0].plot(t_raw, raw_trial.emg[:, c], lw=0.6)
        axes[c, 1].plot(t_proc, proc_trial.emg[:, c], lw=0.6, color="tab:orange")
        axes[c, 0].set_ylabel(f"ch{c + 1}")
    axes[0, 0].set_title("raw")
    axes[0, 1].set_title("signal-processed")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.suptitle(f"signal processing — {tid}")
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
