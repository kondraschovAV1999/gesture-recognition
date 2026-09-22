from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.data.datasets.base import Dataset


def crop_to_gesture(dataset: Dataset, config: GestureCropConfig) -> Dataset:
    """Crop each trial to the gesture region plus a symmetric context margin.

    Gesture trials (gesture_start >= 0) are cropped to
    [gesture_start - pad, gesture_end + pad] (clamped to the trial bounds) and
    their gesture bounds are shifted into the new coordinates. noGesture trials
    (gesture_start < 0) are truncated to no_gesture_max_length_samples (auto =
    2*pad + median gesture length). Trials already shorter than the target are
    left unchanged.
    """
    if not config.enabled:
        return dataset

    pad = config.pad_samples
    gesture_lens = [
        int(g_end) - int(g_start)
        for g_start, g_end in zip(dataset.gesture_start, dataset.gesture_end)
        if int(g_start) >= 0
    ]
    median_len = int(np.median(gesture_lens)) if gesture_lens else None
    max_len = config.no_gesture_max_length_samples
    if max_len is None and median_len is not None:
        max_len = 2 * pad + median_len

    new_trials = []
    new_starts = []
    new_ends = []
    for trial, g_start, g_end in zip(
        dataset.trials, dataset.gesture_start, dataset.gesture_end
    ):
        g_start = int(g_start)
        g_end = int(g_end)
        length = trial.shape[0]
        if g_start < 0:
            # noGesture trial: truncate to max_len when set and longer
            if max_len is not None and length > max_len:
                trial = trial[:max_len]
            new_trials.append(trial)
            new_starts.append(g_start)  # -1
            new_ends.append(g_end)  # -1
        else:
            crop_start = max(0, g_start - pad)
            crop_end = min(length, g_end + pad)
            new_trials.append(trial[crop_start:crop_end])
            new_starts.append(g_start - crop_start)
            new_ends.append(g_end - crop_start)

    return replace(
        dataset,
        trials=new_trials,
        gesture_start=np.asarray(new_starts, dtype=int),
        gesture_end=np.asarray(new_ends, dtype=int),
    )
