from __future__ import annotations

from dataclasses import replace

import numpy as np
from sklearn.base import BaseEstimator

from hand_gesture_recognition.data.datasets.base import Dataset


class BaselineRemoval(BaseEstimator):
    """Subtract a per-channel mean rest level.

    ``mode="global"`` estimates a single per-channel mean from the *train
    partition's* noGesture trials (``gesture_start < 0``) and subtracts it from
    every trial. ``mode="per_user"`` estimates one per-channel mean **per
    subject** from that subject's own rest trials and subtracts each trial's own
    subject's baseline (raising if a subject has no rest trials or no baseline).
    """

    def __init__(self, mode: str = "global"):
        self.mode = mode
        self.baseline_ = None

    def fit(self, train: Dataset) -> "BaselineRemoval":
        if self.mode == "global":
            rest = [
                t for t, gs in zip(train.trials, train.gesture_start) if int(gs) < 0
            ]
            if not rest:
                raise ValueError(
                    "baseline removal requires noGesture (rest) trials in the train partition"
                )
            self.baseline_ = np.mean(np.concatenate(rest, axis=0), axis=0).astype(
                np.float32
            )
        elif self.mode == "per_user":
            self.baseline_ = {}
            for subj in np.unique(train.subject_ids.astype(str)):
                rest = [
                    t
                    for t, gs, s in zip(
                        train.trials, train.gesture_start, train.subject_ids
                    )
                    if str(s) == subj and int(gs) < 0
                ]
                if not rest:
                    raise ValueError(
                        f"subject {subj} has no rest trials in the train partition; "
                        "per_user baseline requires rest trials for every subject"
                    )
                self.baseline_[subj] = np.mean(
                    np.concatenate(rest, axis=0), axis=0
                ).astype(np.float32)
        else:
            raise ValueError(f"unknown baseline mode: {self.mode!r}")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        if self.baseline_ is None:
            raise ValueError("BaselineRemoval must be fit before transform")
        if self.mode == "global":
            trials = [t.astype(np.float32) - self.baseline_ for t in dataset.trials]
        else:
            trials = []
            for t, s in zip(dataset.trials, dataset.subject_ids):
                b = self.baseline_.get(str(s))
                if b is None:
                    raise ValueError(
                        f"no baseline for subject {s}; per_user baseline requires "
                        "the subject to appear in train"
                    )
                trials.append(t.astype(np.float32) - b)
        return replace(dataset, trials=trials)
