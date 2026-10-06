from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset


def build_no_gesture_remap(dataset: Dataset) -> Tuple[Dict[int, int], List[str]]:
    """Compute the global label remap and reduced class_names (no noGesture).

    Computed ONCE from the full (pre-split) dataset so every partition applies
    the same mapping, even if a partition is missing a class.
    """
    keep = dataset.gesture_start >= 0
    kept_labels = np.unique(dataset.labels[keep])
    remap = {int(old): new for new, old in enumerate(kept_labels)}
    new_class_names = [dataset.metadata.class_names[int(o)] for o in kept_labels]
    return remap, new_class_names


def drop_no_gesture(
    dataset: Dataset, remap: Dict[int, int], new_class_names: List[str]
) -> Dataset:
    """Drop rest trials and remap labels using a GLOBAL remap (per-partition)."""
    keep = dataset.gesture_start >= 0
    new_labels = np.asarray([remap[int(l)] for l in dataset.labels[keep]], dtype=int)
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


def remove_no_gesture(dataset: Dataset) -> Dataset:
    """Convenience: build the global remap and apply it to a single dataset."""
    remap, new_class_names = build_no_gesture_remap(dataset)
    return drop_no_gesture(dataset, remap, new_class_names)
