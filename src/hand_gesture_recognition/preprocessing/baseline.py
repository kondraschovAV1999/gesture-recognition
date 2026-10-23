from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional, Union, cast

import numpy as np
from sklearn.base import BaseEstimator

from hand_gesture_recognition.data.datasets.base import Dataset, Trial


class BaselineRemoval(BaseEstimator):
    """Subtract a per-channel mean rest level.

    ``mode="global"`` estimates a single per-channel mean from the *train
    partition's* noGesture trials (``trial.is_rest``, i.e. ``start < 0``) and
    subtracts it from every trial. ``mode="per_user"`` estimates one per-channel
    mean **per subject** from that subject's own rest trials and subtracts each
    trial's own subject's baseline (raising if a subject has no rest trials or
    no baseline).
    """

    def __init__(self, mode: str = "global"):
        self.mode = mode
        self.baseline_: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None

    def fit(self, train: Dataset) -> "BaselineRemoval":
        if self.mode == "global":
            rest = [t.emg for t in train.trials if t.is_rest]
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
                    t.emg for t in train.trials if str(t.user_id) == subj and t.is_rest
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
            baseline = cast(np.ndarray, self.baseline_)
            new_trials = [
                replace(
                    t,
                    emg=(t.emg.astype(np.float32) - baseline).astype(np.float32),
                )
                for t in dataset.trials
            ]
            return dataset.replace_trials(new_trials)

        baselines = cast(Dict[str, np.ndarray], self.baseline_)
        new_trials = []
        for t in dataset.trials:
            b = baselines.get(str(t.user_id))
            if b is None:
                raise ValueError(
                    f"no baseline for subject {t.user_id}; per_user baseline "
                    "requires the subject to appear in train"
                )
            new_trials.append(
                replace(t, emg=(t.emg.astype(np.float32) - b).astype(np.float32))
            )
        return dataset.replace_trials(new_trials)
