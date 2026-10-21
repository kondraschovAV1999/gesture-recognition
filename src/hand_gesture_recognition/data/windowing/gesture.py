from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowMetadata,
)


def window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    """Build one sample per gesture trial (classification formulation).

    Each gesture trial (``gesture_start >= 0``) contributes exactly one window:
    the segment ``trial[gesture_start:gesture_end]`` (the whole gesture, no
    context margin), zero-padded at the end if shorter than the window size and
    truncated to the first ``window_size`` samples if longer. Rest trials
    (``gesture_start < 0``) are dropped entirely. Returns the same
    ``WindowedDataset`` shape as ``window_dataset`` so all downstream stages
    (classical/neural backends, evaluation, reporting) are unchanged.
    """
    ws, _ = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
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
        g_start = int(g_start)
        g_end = int(g_end)
        if g_start < 0:
            continue  # rest trial: dropped in classification mode
        seg = trial[g_start:g_end]
        if seg.shape[0] < ws:
            pad = np.zeros((ws - seg.shape[0], seg.shape[1]), dtype=seg.dtype)
            seg = np.vstack([seg, pad])
        elif seg.shape[0] > ws:
            seg = seg[:ws]
        xs.append(seg)
        ys.append(int(label))
        starts.append(g_start)
        ends.append(g_end)
        subjects.append(str(subj))
        trials.append(str(tid))
        wins.append(f"{subj}_{tid}_gesture")

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
