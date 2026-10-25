from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.configuration.schema import (
    Dataset,
    WindowingConfig,
    WindowMetadata,
    WindowedDataset,
)


def _resolve_samples(config):
    if config.window_size_samples is None:
        raise ValueError("window_emg requires window_size_samples")
    stride = config.stride_samples or config.window_size_samples
    return config.window_size_samples, stride


def _empty_meta() -> WindowMetadata:
    return WindowMetadata(
        window_start=np.asarray([], dtype=int),
        window_end=np.asarray([], dtype=int),
        subject_ids=np.asarray([], dtype=object),
        trial_ids=np.asarray([], dtype=object),
        window_ids=np.asarray([], dtype=object),
    )


def _concat_meta(metas) -> WindowMetadata:
    if not metas:
        return _empty_meta()
    return WindowMetadata(
        window_start=np.concatenate([m.window_start for m in metas]),
        window_end=np.concatenate([m.window_end for m in metas]),
        subject_ids=np.concatenate([m.subject_ids for m in metas]),
        trial_ids=np.concatenate([m.trial_ids for m in metas]),
        window_ids=np.concatenate([m.window_ids for m in metas]),
    )


def window_emg(
    emg, label, config, subject_id, trial_id, prefix, rest_bounds=None, drop_rest=False
):
    ws, stride = _resolve_samples(config)
    emg = np.asarray(emg, dtype=np.float32)
    length = emg.shape[0]
    if length < ws:
        if config.padding == "zero":
            pad = np.zeros((ws - length, emg.shape[1]), dtype=emg.dtype)
            emg = np.vstack([emg, pad])
            length = ws
        else:
            return (
                np.zeros((0, ws, emg.shape[1]), np.float32),
                np.array([], int),
                _empty_meta(),
            )
    g_start = g_end = None
    if rest_bounds is not None:
        g_start, g_end = rest_bounds
    n = (length - ws) // stride + 1
    xs, ys, starts, ends, wins = [], [], [], [], []
    for i in range(n):
        s = i * stride
        e = s + ws
        if rest_bounds is None:
            is_rest = False
        else:
            is_rest = (g_start < 0) or (e <= g_start) or (s >= g_end)
        if drop_rest and is_rest:
            continue
        xs.append(emg[s:e])
        ys.append(0 if is_rest else int(label))
        starts.append(s)
        ends.append(e)
        wins.append(f"{subject_id}_{trial_id}_{prefix}{i}")
    meta = WindowMetadata(
        window_start=np.asarray(starts, int),
        window_end=np.asarray(ends, int),
        subject_ids=np.asarray([subject_id] * len(xs)),
        trial_ids=np.asarray([trial_id] * len(xs)),
        window_ids=np.asarray(wins),
    )
    X = np.stack(xs) if xs else np.zeros((0, ws, emg.shape[1]), np.float32)
    return X, np.asarray(ys, int), meta


def window_dataset(dataset: Dataset, config: WindowingConfig) -> WindowedDataset:
    ws, stride = config.resolve(sampling_rate=dataset.metadata.sampling_rate)
    resolved = replace(config, window_size_samples=ws, stride_samples=stride)
    # After no-gesture is dropped, class_names no longer contains no_gesture_name;
    # in that case rest windows (label 0) would collide with a real gesture, so
    # they must be dropped. This replaces the old drop_rest_windows metadata flag.
    drop_rest = dataset.metadata.no_gesture_name not in dataset.metadata.class_names
    Xs, ys, metas = [], [], []
    for t in dataset.trials:
        X, y, m = window_emg(
            t.emg,
            dataset.metadata.class_names.index(t.gesture_name),
            resolved,
            t.user_id,
            t.trial_id,
            "window",
            rest_bounds=(t.start, t.end),  # rest trial -> (-1,-1) -> all rest
            drop_rest=drop_rest,
        )
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        metas.append(m)
    if not Xs:
        n_ch = (
            len(dataset.metadata.channels)
            if dataset.metadata.channels
            else (dataset.trials[0].emg.shape[1] if dataset.trials else 0)
        )
        return WindowedDataset(
            X=np.zeros((0, ws, n_ch), np.float32),
            y=np.asarray([], int),
            metadata=_empty_meta(),
        )
    return WindowedDataset(
        X=np.concatenate(Xs, axis=0),
        y=np.concatenate(ys, axis=0).astype(int),
        metadata=_concat_meta(metas),
    )
