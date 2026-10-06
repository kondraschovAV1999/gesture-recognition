from __future__ import annotations

from dataclasses import replace

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def remove_no_gesture(dataset: Dataset) -> Dataset:
    """Remove the rest (no-gesture) class from a dataset.

    Rest trials are exactly those with ``gesture_start < 0``. They are dropped,
    the remaining labels are remapped to a contiguous ``0..N-1`` range (in
    ascending order of the original labels), ``metadata.class_names`` is reduced
    to the kept classes, and ``metadata.drop_rest_windows`` is set so windowing
    later *skips* (rather than labels) the rest windows outside each gesture's
    bounds. If there are no rest trials (e.g. the synthetic dataset), the
    dataset is returned unchanged.
    """
    keep = dataset.gesture_start >= 0
    if keep.all():
        return dataset

    kept_labels = np.unique(dataset.labels[keep])
    old_to_new = {int(old): new for new, old in enumerate(kept_labels)}
    new_labels = np.asarray(
        [old_to_new[int(l)] for l in dataset.labels[keep]], dtype=int
    )
    new_class_names = [dataset.metadata.class_names[int(o)] for o in kept_labels]

    remaining_subjects = set(str(s) for s in dataset.subject_ids[keep])
    subject_info = dataset.subject_info
    if subject_info is not None:
        subject_info = subject_info.loc[subject_info.index.isin(remaining_subjects)]

    return replace(
        dataset,
        trials=[t for t, k in zip(dataset.trials, keep) if k],
        labels=new_labels,
        subject_ids=dataset.subject_ids[keep],
        trial_ids=dataset.trial_ids[keep],
        gesture_start=dataset.gesture_start[keep],
        gesture_end=dataset.gesture_end[keep],
        subject_info=subject_info,
        metadata=replace(
            dataset.metadata,
            class_names=new_class_names,
            drop_rest_windows=True,
        ),
    )
