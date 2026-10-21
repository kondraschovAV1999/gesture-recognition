from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowMetadata,
)


@dataclass
class TwoLevelWindowedDataset:
    big_windows: np.ndarray  # (n_big, big_size, channels) — signal-processed
    small_windows: List[np.ndarray]  # per big window: (n_small_i, small_size, channels)
    y: np.ndarray  # (n_big,) big-window labels
    metadata: WindowMetadata  # big-window level


@dataclass
class TwoLevelWindowedPartitions:
    train: TwoLevelWindowedDataset
    val: TwoLevelWindowedDataset
    test: TwoLevelWindowedDataset


def window_dataset_two_level(
    dataset: Dataset,
    config: WindowingConfig,
    signal_steps: List[Any],
) -> TwoLevelWindowedDataset:
    """Slide big windows, filter each, slide small windows inside, label big.

    ``signal_steps`` are already-instantiated transform objects (each with a
    ``transform(segment)`` method), applied per big window.
    """
    big_size, big_stride = config.resolve_big(
        sampling_rate=dataset.metadata.sampling_rate
    )
    small_size, small_stride = config.resolve(
        sampling_rate=dataset.metadata.sampling_rate
    )
    drop_rest = dataset.metadata.drop_rest_windows
    n_channels = (
        len(dataset.metadata.channels)
        if dataset.metadata.channels
        else (dataset.trials[0].shape[1] if dataset.trials else 0)
    )

    big_windows: List[np.ndarray] = []
    small_windows: List[np.ndarray] = []
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
        length = trial.shape[0]
        if length < big_size:
            if config.padding == "zero":
                pad = np.zeros((big_size - length, trial.shape[1]), dtype=trial.dtype)
                trial = np.vstack([trial, pad])
                length = big_size
            else:
                continue  # drop trials shorter than a big window
        n_big = (length - big_size) // big_stride + 1
        for i in range(n_big):
            s = i * big_stride
            e = s + big_size
            # A big window is "rest" if it does not overlap [g_start, g_end]
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
            if drop_rest and is_rest:
                continue
            big_seg = trial[s:e]
            for step in signal_steps:
                big_seg = step.transform(big_seg)
            blen = big_seg.shape[0]
            smalls = []
            if blen >= small_size:
                n_small = (blen - small_size) // small_stride + 1
                for j in range(n_small):
                    ss = j * small_stride
                    smalls.append(big_seg[ss : ss + small_size])
            if not smalls:
                continue  # defensive: no small window fits
            big_windows.append(big_seg)
            small_windows.append(np.stack(smalls, axis=0))
            ys.append(0 if is_rest else int(label))
            starts.append(s)
            ends.append(e)
            subjects.append(str(subj))
            trials.append(str(tid))
            wins.append(f"{subj}_{tid}_big{i}")

    return TwoLevelWindowedDataset(
        big_windows=np.stack(big_windows, axis=0)
        if big_windows
        else np.zeros((0, big_size, n_channels), dtype=np.float32),
        small_windows=small_windows,
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.asarray(starts, dtype=int),
            window_end=np.asarray(ends, dtype=int),
            subject_ids=np.asarray(subjects),
            trial_ids=np.asarray(trials),
            window_ids=np.asarray(wins),
        ),
    )


def window_big_windows(
    dataset: Dataset,
    config: WindowingConfig,
    signal_steps: List[Any],
) -> WindowedDataset:
    """Flat big-window view (for the classical path and prediction overlays)."""
    twd = window_dataset_two_level(dataset, config, signal_steps)
    return WindowedDataset(X=twd.big_windows, y=twd.y, metadata=twd.metadata)
