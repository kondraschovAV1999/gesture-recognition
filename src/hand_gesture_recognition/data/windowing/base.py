from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset


@dataclass
class WindowMetadata:
    window_start: np.ndarray  # (n_windows,)
    window_end: np.ndarray  # (n_windows,)
    subject_ids: np.ndarray  # (n_windows,)
    trial_ids: np.ndarray  # (n_windows,)


@dataclass
class WindowedDataset:
    X: np.ndarray  # (n_windows, window_size, channels)
    y: np.ndarray  # (n_windows,)
    metadata: WindowMetadata


def window_dataset(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    ws, stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    subjects: List[str] = []
    trials: List[str] = []

    for trial, label, subj, tid, g_start, g_end in zip(
        dataset.trials,
        dataset.labels,
        dataset.subject_ids,
        dataset.trial_ids,
        dataset.gesture_start,
        dataset.gesture_end,
    ):
        length = trial.shape[0]
        if length < ws:
            if config.padding == "zero":
                pad = np.zeros((ws - length, trial.shape[1]), dtype=trial.dtype)
                trial = np.vstack([trial, pad])
                length = ws
            else:
                continue  # drop trials shorter than a window (config.padding == "none")
        n_windows = (length - ws) // stride + 1
        for i in range(n_windows):
            s = i * stride
            e = s + ws
            # Per-window label from the gesture ground-truth bounds: 0 (rest)
            # outside [start, end], the trial's gesture label (1..5) inside.
            if g_start < 0:  # rest trial
                w_label = 0
            elif e <= g_start:  # window fully before gesture
                w_label = 0
            elif s >= g_end:  # window fully after gesture
                w_label = 0
            else:  # window overlaps the gesture segment
                w_label = int(label)
            xs.append(trial[s:e])
            ys.append(w_label)
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))

    return WindowedDataset(
        X=np.stack(xs, axis=0)
        if xs
        else np.zeros((0, ws, dataset.trials[0].shape[1]), dtype=np.float32),
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
        ),
    )
