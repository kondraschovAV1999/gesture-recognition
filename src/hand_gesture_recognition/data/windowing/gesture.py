from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowMetadata,
    _concat_meta,
)


def window_gestures(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    """Build one sample per gesture trial (classification formulation).

    Each non-rest trial (``is_rest`` false) contributes exactly one window: the
    segment ``trial.emg[trial.start:trial.end]`` (the whole gesture, no context
    margin), zero-padded at the end if shorter than the window size and
    truncated to the first ``ws`` samples if longer. Rest trials are dropped
    entirely. Returns the same ``WindowedDataset`` shape as ``window_dataset``
    so all downstream stages (classical/neural backends, evaluation, reporting)
    are unchanged.
    """
    ws, _ = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    metas: List[WindowMetadata] = []

    for t in dataset.trials:
        if t.is_rest:
            continue  # rest trial: dropped in classification mode
        seg = t.emg[t.start : t.end]
        if seg.shape[0] < ws:
            pad = np.zeros((ws - seg.shape[0], seg.shape[1]), dtype=seg.dtype)
            seg = np.vstack([seg, pad])
        elif seg.shape[0] > ws:
            seg = seg[:ws]
        xs.append(seg)
        ys.append(dataset.metadata.class_names.index(t.gesture_name))
        metas.append(
            WindowMetadata(
                window_start=np.asarray([t.start], int),
                window_end=np.asarray([t.end], int),
                subject_ids=np.asarray([t.user_id]),
                trial_ids=np.asarray([t.trial_id]),
                window_ids=np.asarray([f"{t.user_id}_{t.trial_id}_gesture"]),
            )
        )

    if not xs:
        n_ch = (
            len(dataset.metadata.channels)
            if dataset.metadata.channels
            else (dataset.trials[0].emg.shape[1] if dataset.trials else 0)
        )
        return WindowedDataset(
            X=np.zeros((0, ws, n_ch), dtype=np.float32),
            y=np.asarray([], dtype=int),
            metadata=_concat_meta([]),
        )

    return WindowedDataset(
        X=np.stack(xs, axis=0),
        y=np.asarray(ys, dtype=int),
        metadata=_concat_meta(metas),
    )
