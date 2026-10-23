from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)


def make_synthetic_dataset(
    n_subjects: int = 4,
    n_trials_per_subject: int = 6,
    n_classes: int = 3,
    length: int = 100,
    n_channels: int = 8,
    seed: int = 0,
) -> Dataset:
    rng = np.random.default_rng(seed)
    class_names = [f"class_{i}" for i in range(n_classes)]
    users: List[User] = []
    for s in range(n_subjects):
        user_trials = []
        for t in range(n_trials_per_subject):
            label = (s + t) % n_classes
            base = rng.normal(0, 1, size=(length, n_channels))
            signal = (base + label).astype(np.float32)
            user_trials.append(
                Trial(
                    trial_id=f"s{s}_t{t}",
                    user_id=f"s{s}",
                    gesture_name=class_names[label],
                    start=0,
                    end=length,
                    emg=signal,
                )
            )
        users.append(User(f"s{s}", user_trials))
    return Dataset(
        metadata=DatasetMetadata(
            name="synthetic",
            version="0",
            sampling_rate=200.0,
            channels=[f"ch{i}" for i in range(1, n_channels + 1)],
            class_names=class_names,
            content_hash="synthetic",
            provenance={"seed": seed},
        ),
        users=users,
    )
