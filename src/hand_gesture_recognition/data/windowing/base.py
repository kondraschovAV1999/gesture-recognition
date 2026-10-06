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
    window_ids: np.ndarray  # (n_windows,) unique, traceable to trial


@dataclass
class WindowedDataset:
    X: np.ndarray  # (n_windows, window_size, channels)
    y: np.ndarray  # (n_windows,)
    metadata: WindowMetadata


@dataclass
class WindowedPartitions:
    """The single authoritative windowed representation of train/val/test."""

    train: WindowedDataset
    val: WindowedDataset
    test: WindowedDataset


def window_dataset(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    ws, stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    drop_rest = dataset.metadata.drop_rest_windows
    n_channels = (
        len(dataset.metadata.channels)
        if dataset.metadata.channels
        else (dataset.trials[0].shape[1] if dataset.trials else 0)
    )
    xs: List[np.ndarray] = []
    ys: List[int] = []
    starts: List[int] = []
    ends: List[int] = []
    subjects: List[str] = []
    trials: List[str] = []
    wins: List[str] = []

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
            # A window is "rest" if it does not overlap the gesture segment
            # [g_start, g_end] (or the trial itself is a rest trial, g_start < 0).
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
            if drop_rest and is_rest:
                continue  # no-gesture class removed: skip rest windows entirely
            w_label = 0 if is_rest else int(label)
            xs.append(trial[s:e])
            ys.append(w_label)
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))
            wins.append(f"{subj}_{tid}_window{i}")

    return WindowedDataset(
        X=np.stack(xs, axis=0)
        if xs
        else np.zeros((0, ws, n_channels), dtype=np.float32),
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
            window_ids=np.asarray(wins),
        ),
    )
