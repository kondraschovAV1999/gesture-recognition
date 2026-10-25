from __future__ import annotations

from dataclasses import replace
from typing import Dict, List

import numpy as np

from hand_gesture_recognition.configuration.schema import (
    GroupedWindowedDataset,
    WindowMetadata,
    WindowedDataset,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import (
    _empty_meta,
    window_dataset,
    window_emg,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.gesture import (
    window_gestures,
)

_REGISTRY: Dict[str, type] = {}


def register_windowing(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_windowing_strategy(name):
    return _REGISTRY[name]()


def _big_config(config, sampling_rate):
    big_ws, big_stride = config.resolve_big(sampling_rate)
    return replace(config, window_size_samples=big_ws, stride_samples=big_stride)


def _build_windowed(xs, ys, metas, config):
    if not xs:
        return WindowedDataset(
            X=np.zeros((0, 0, 0), np.float32),
            y=np.asarray([], int),
            metadata=_empty_meta(),
        )
    X = np.stack(xs)
    return WindowedDataset(
        X=X,
        y=np.asarray(ys, dtype=int),
        metadata=WindowMetadata(
            window_start=np.concatenate([m.window_start for m in metas]),
            window_end=np.concatenate([m.window_end for m in metas]),
            subject_ids=np.concatenate([m.subject_ids for m in metas]),
            trial_ids=np.concatenate([m.trial_ids for m in metas]),
            window_ids=np.concatenate([m.window_ids for m in metas]),
        ),
    )


class RecognitionWindowing:
    def window(self, dataset, config, signal_steps=None):
        return window_dataset(dataset, config)


class ClassificationWindowing:
    def window(self, dataset, config, signal_steps=None):
        return window_gestures(dataset, config)


class TwoLevelWindowing:
    def window(self, dataset, config, signal_steps=None):
        signal_steps = signal_steps or []
        big_cfg = _big_config(config, dataset.metadata.sampling_rate)
        small_ws, small_stride = config.resolve(dataset.metadata.sampling_rate)
        small_cfg = replace(
            config, window_size_samples=small_ws, stride_samples=small_stride
        )
        big_X, big_y, big_metas = [], [], []
        smalls, big_ids = [], []
        for t in dataset.trials:
            X, y, m = window_emg(
                t.emg,
                dataset.metadata.class_names.index(t.gesture_name),
                big_cfg,
                t.user_id,
                t.trial_id,
                "big",
                rest_bounds=(t.start, t.end),  # rest trial -> (-1,-1) -> all rest
                drop_rest=(
                    dataset.metadata.no_gesture_name not in dataset.metadata.class_names
                ),
            )
            for j in range(len(y)):
                big = X[j]
                for step in signal_steps:
                    big = step.transform(big)
                sX, _sy, _sm = window_emg(
                    big,
                    int(y[j]),
                    small_cfg,
                    t.user_id,
                    t.trial_id,
                    f"big{j}_",
                    rest_bounds=None,
                    drop_rest=False,
                )
                smalls.append(sX)
                big_ids.append(m.window_ids[j])
                big_X.append(big)
                big_y.append(int(y[j]))
                # per-big-window metadata: take the j-th row of m
                big_metas.append(_row_meta(m, j))
        big_wd = _build_windowed(big_X, big_y, big_metas, config)
        return GroupedWindowedDataset(
            big=big_wd,
            small_windows=smalls,
            big_window_id=np.asarray(big_ids),
            y=np.asarray(big_y, dtype=int),
            metadata=big_wd.metadata,
        )


def _row_meta(m, j):
    return WindowMetadata(
        window_start=m.window_start[j : j + 1],
        window_end=m.window_end[j : j + 1],
        subject_ids=m.subject_ids[j : j + 1],
        trial_ids=m.trial_ids[j : j + 1],
        window_ids=m.window_ids[j : j + 1],
    )


register_windowing("recognition")(RecognitionWindowing)
register_windowing("classification")(ClassificationWindowing)
register_windowing("two_level")(TwoLevelWindowing)
