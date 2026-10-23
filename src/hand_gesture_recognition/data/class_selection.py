from __future__ import annotations

from hand_gesture_recognition.data.datasets.base import Dataset


def build_no_gesture_remap(dataset: Dataset):
    kept = [t for t in dataset.trials if not t.is_rest]
    kept_names = []
    for t in kept:
        if t.gesture_name not in kept_names:
            kept_names.append(t.gesture_name)
    remap = {dataset.metadata.class_names.index(n): i for i, n in enumerate(kept_names)}
    return remap, kept_names


def drop_no_gesture(dataset, remap, new_class_names):
    from dataclasses import replace

    kept = [t for t in dataset.trials if not t.is_rest]
    new_meta = replace(dataset.metadata, class_names=new_class_names)
    return Dataset(metadata=new_meta, users=Dataset.from_trials(kept, new_meta).users)


def remove_no_gesture(dataset):
    remap, names = build_no_gesture_remap(dataset)
    return drop_no_gesture(dataset, remap, names)
