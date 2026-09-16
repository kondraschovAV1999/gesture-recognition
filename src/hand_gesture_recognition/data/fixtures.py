from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata


def make_synthetic_dataset(
    n_subjects: int = 4,
    n_trials_per_subject: int = 6,
    n_classes: int = 3,
    length: int = 100,
    n_channels: int = 8,
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    trials: List[np.ndarray] = []
    labels: List[int] = []
    subject_ids: List[str] = []
    trial_ids: List[str] = []
    for s in range(n_subjects):
        for t in range(n_trials_per_subject):
            label = (s + t) % n_classes
            base = rng.normal(0, 1, size=(length, n_channels))
            signal = base + label  # class-separable offset
            trials.append(signal.astype(np.float32))
            labels.append(label)
            subject_ids.append(f"s{s}")
            trial_ids.append(f"s{s}_t{t}")
    return Dataset(
        trials=trials,
        labels=np.asarray(labels, dtype=int),
        subject_ids=np.asarray(subject_ids),
        trial_ids=np.asarray(trial_ids),
        gesture_start=np.zeros(len(trials), dtype=int),
        gesture_end=np.array([t.shape[0] for t in trials], dtype=int),
        metadata=DatasetMetadata(
            name="synthetic",
            version="0",
            sampling_rate=200.0,
            channels=[f"ch{i}" for i in range(1, n_channels + 1)],
            class_names=[f"class_{i}" for i in range(n_classes)],
            content_hash="synthetic",
            provenance={"seed": seed},
        ),
    )
