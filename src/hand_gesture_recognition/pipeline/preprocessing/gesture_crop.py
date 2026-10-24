from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.utils.datasets import Dataset


def crop_to_gesture(dataset: Dataset, config: GestureCropConfig) -> Dataset:
    """Crop each trial to the gesture region plus a symmetric context margin.

    Gesture trials (start >= 0) are cropped to [start - pad, end + pad]
    (clamped to the trial bounds) and their gesture bounds are shifted into the
    new coordinates. noGesture trials (start < 0) are truncated to
    no_gesture_max_length_samples (auto = 2*pad + median gesture length). Trials
    already shorter than the target are left unchanged.
    """
    if not config.enabled:
        return dataset

    pad = config.pad_samples
    gesture_lens = [
        int(t.end) - int(t.start) for t in dataset.trials if int(t.start) >= 0
    ]
    median_len = int(np.median(gesture_lens)) if gesture_lens else None
    max_len = config.no_gesture_max_length_samples
    if max_len is None and median_len is not None:
        max_len = 2 * pad + median_len

    new_trials = []
    for trial in dataset.trials:
        start = int(trial.start)
        end = int(trial.end)
        emg = trial.emg
        length = emg.shape[0]
        if start < 0:
            # noGesture trial: truncate to max_len when set and longer
            if max_len is not None and length > max_len:
                emg = emg[:max_len]
            new_trials.append(replace(trial, emg=emg))
        else:
            crop_start = max(0, start - pad)
            crop_end = min(length, end + pad)
            new_trials.append(
                replace(
                    trial,
                    emg=emg[crop_start:crop_end],
                    start=start - crop_start,
                    end=end - crop_start,
                )
            )

    return dataset.replace_trials(new_trials)
