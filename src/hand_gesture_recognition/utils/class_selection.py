from __future__ import annotations

from hand_gesture_recognition.configuration.schema import Dataset


def build_no_gesture_remap(dataset: Dataset):
    kept = [t for t in dataset.trials if not t.is_rest]
    # Sort kept names by their original index in the full class_names list,
    # matching the old np.unique(labels) ordering (deterministic, order-preserving).
    kept_names = sorted(
        {t.gesture_name for t in kept},
        key=dataset.metadata.class_names.index,
    )
    remap = {dataset.metadata.class_names.index(n): i for i, n in enumerate(kept_names)}
    return remap, kept_names


def drop_no_gesture(dataset, remap, new_class_names):
    from dataclasses import replace

    # ``remap`` is unused: labels re-derive from ``class_names`` on the rebuilt
    # dataset (see Dataset.labels). Kept in the signature to avoid rippling into
    # the Pipeline.
    kept = [t for t in dataset.trials if not t.is_rest]
    new_meta = replace(dataset.metadata, class_names=new_class_names)
    return dataset.replace_trials(kept, metadata=new_meta)


def remove_no_gesture(dataset):
    remap, names = build_no_gesture_remap(dataset)
    return drop_no_gesture(dataset, remap, names)
