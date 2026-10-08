from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.splits.base import (
    DataSplit,
    _subset,
    assert_no_group_leakage,
)


def _group_ids(dataset: Dataset, split: SplitConfig) -> np.ndarray:
    return dataset.subject_ids if split.split_on == "subject_ids" else dataset.trial_ids


def _carve_val_group(
    train_idx: np.ndarray, groups: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a subject-wise val subset out of train_idx (group-level)."""
    rng = np.random.default_rng(seed)
    train_groups = np.unique(groups[train_idx])
    rng.shuffle(train_groups)
    n_val = max(1, int(round(len(train_groups) * val_ratio)))
    val_groups = set(train_groups[:n_val].tolist())
    val_idx = np.asarray([i for i in train_idx if groups[i] in val_groups], dtype=int)
    new_train = np.asarray(
        [i for i in train_idx if groups[i] not in val_groups], dtype=int
    )
    return new_train, val_idx


def _carve_val_trial(
    train_idx: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a trial-wise val subset out of train_idx."""
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def _empty(dataset: Dataset) -> Dataset:
    return _subset(dataset, np.asarray([], dtype=int))


def generate_folds(
    dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig
) -> List[DataSplit]:
    """Produce the list of DataSplits for a cross-validation run."""
    cv.validate()
    n = dataset.n_trials
    indices = np.arange(n)
    folds: List[DataSplit] = []

    def carve(train_idx: np.ndarray):
        if not cv.include_val:
            return train_idx, np.asarray([], dtype=int)
        if cv.method == "loso" or cv.group_folds:
            return _carve_val_group(
                train_idx, _group_ids(dataset, split), cv.val_ratio, cv.seed
            )
        return _carve_val_trial(train_idx, cv.val_ratio, cv.seed)

    if cv.method == "loso":
        groups = _group_ids(dataset, split)
        for g in np.unique(groups):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
        return folds

    # kfold
    if cv.group_folds:
        groups = _group_ids(dataset, split)
        splitter = GroupKFold(n_splits=cv.n_splits)
        for train_idx, test_idx in splitter.split(indices, groups=groups):
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
    else:
        splitter = KFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.seed)
        for train_idx, test_idx in splitter.split(indices):
            train_idx, val_idx = carve(train_idx)
            folds.append(
                DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
            )
    return folds
