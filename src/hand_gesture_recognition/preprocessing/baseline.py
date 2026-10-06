from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator

from hand_gesture_recognition.data.datasets.base import Dataset


class BaselineRemoval(BaseEstimator):
    """Subtract the global per-channel mean rest level.

    ``fit`` estimates the baseline from the *train partition's* noGesture
    trials only (``gesture_start < 0``); ``transform`` subtracts the frozen
    per-channel mean from every trial of any partition.
    """

    def __init__(self):
        self.baseline_ = None

    def fit(self, train: Dataset) -> "BaselineRemoval":
        rest = [t for t, gs in zip(train.trials, train.gesture_start) if int(gs) < 0]
        if not rest:
            raise ValueError(
                "baseline removal requires noGesture (rest) trials in the train partition"
            )
        self.baseline_ = np.mean(np.concatenate(rest, axis=0), axis=0).astype(
            np.float32
        )
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if self.baseline_ is None:
            raise ValueError("BaselineRemoval must be fit before transform")
        trials = [t.astype(np.float32) - self.baseline_ for t in dataset.trials]
        return replace(dataset, trials=trials)
