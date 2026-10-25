from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.configuration.schema import Dataset
from hand_gesture_recognition.configuration.schema import DataSplit
from hand_gesture_recognition.pipeline.splits.base import (
    _subset,
    assert_no_group_leakage,
)

_REGISTRY: dict = {}


def register_split(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls

    return deco


def get_split_strategy(name):
    return _REGISTRY[name]()


class RandomSplit:
    """Trial-level split: a random permutation of all trials.

    Handles both ``random`` and ``trial_wise``, which share the same body.
    """

    def split(self, dataset, config) -> DataSplit:
        n = dataset.n_trials
        rng = np.random.default_rng(config.seed)
        perm = rng.permutation(n)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        train_idx, val_idx, test_idx = (
            perm[:n_train],
            perm[n_train : n_train + n_val],
            perm[n_train + n_val :],
        )
        return DataSplit(
            train=_subset(dataset, train_idx),
            val=_subset(dataset, val_idx),
            test=_subset(dataset, test_idx),
        )


class StratifiedSplit:
    def split(self, dataset, config) -> DataSplit:
        from sklearn.model_selection import train_test_split

        n = dataset.n_trials
        train_idx, rest_idx = train_test_split(
            np.arange(n),
            train_size=config.train_ratio,
            stratify=dataset.labels,
            random_state=config.seed,
        )
        val_ratio = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_idx, test_idx = train_test_split(
            rest_idx,
            test_size=1 - val_ratio,
            stratify=dataset.labels[rest_idx],
            random_state=config.seed,
        )
        return DataSplit(
            train=_subset(dataset, train_idx),
            val=_subset(dataset, val_idx),
            test=_subset(dataset, test_idx),
        )


class GroupWiseSplit:
    def split(self, dataset, config) -> DataSplit:
        config.validate()
        groups = (
            dataset.subject_ids
            if config.split_on == "subject_ids"
            else dataset.trial_ids
        )
        n = dataset.n_trials
        indices = np.arange(n)

        # Reserve test groups first (group-wise, seeded).
        test_splitter = GroupShuffleSplit(
            n_splits=1, test_size=config.test_ratio, random_state=config.seed
        )
        train_val_idx, test_idx = next(test_splitter.split(indices, groups=groups))

        # Then carve a validation subset (group-wise) out of the remaining train+val.
        remaining_groups = groups[train_val_idx]
        relative_val_ratio = config.val_ratio / (config.train_ratio + config.val_ratio)
        val_splitter = GroupShuffleSplit(
            n_splits=1, test_size=relative_val_ratio, random_state=config.seed
        )
        train_rel_idx, val_rel_idx = next(
            val_splitter.split(train_val_idx, groups=remaining_groups)
        )
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        return DataSplit(
            train=_subset(dataset, train_idx),
            val=_subset(dataset, val_idx),
            test=_subset(dataset, test_idx),
        )


class SessionWiseSplit(GroupWiseSplit):
    """Session-wise split. Behaves identically to ``group_wise`` today (the
    same body) because Phase-1 datasets lack a session-bearing field."""


class PredefinedSplit:
    def split(self, dataset, config) -> DataSplit:
        # Subject-independent split: train/val = subject-wise split of the train
        # cohort (trainingJSON); test = the test cohort (testingJSON), held out
        # entirely. Gives a genuinely independent user-level test (no subject
        # appears in more than one partition).
        rng = np.random.default_rng(config.seed)
        sids = dataset.subject_ids.astype(str)
        is_train = np.char.startswith(sids, "train:")
        is_test = np.char.startswith(sids, "test:")

        train_subjects = np.unique(dataset.subject_ids[is_train])
        rng.shuffle(train_subjects)
        n_val = max(1, int(round(len(train_subjects) * config.val_ratio)))
        val_subjects = list(train_subjects[:n_val].tolist())

        train_idx = np.where(is_train & ~np.isin(dataset.subject_ids, val_subjects))[0]
        val_idx = np.where(is_train & np.isin(dataset.subject_ids, val_subjects))[0]
        test_idx = np.where(is_test)[0]
        return DataSplit(
            train=_subset(dataset, train_idx),
            val=_subset(dataset, val_idx),
            test=_subset(dataset, test_idx),
        )


register_split("random")(RandomSplit)
register_split("trial_wise")(RandomSplit)
register_split("stratified")(StratifiedSplit)
register_split("group_wise")(GroupWiseSplit)
register_split("session_wise")(SessionWiseSplit)
register_split("predefined")(PredefinedSplit)


_CV_REGISTRY: dict = {}


def register_cv(name):
    def deco(cls):
        _CV_REGISTRY[name] = cls
        return cls

    return deco


def get_cv_strategy(name):
    return _CV_REGISTRY[name]()


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


def _carve(
    dataset: Dataset,
    cv: CrossValidationConfig,
    split: SplitConfig,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not cv.include_val:
        return train_idx, np.asarray([], dtype=int)
    if cv.method == "loso" or cv.group_folds:
        return _carve_val_group(
            train_idx, _group_ids(dataset, split), cv.val_ratio, cv.seed
        )
    return _carve_val_trial(train_idx, cv.val_ratio, cv.seed)


class KFoldCVStrategy:
    def generate(
        self, dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig
    ) -> List[DataSplit]:
        n = dataset.n_trials
        indices = np.arange(n)
        folds: List[DataSplit] = []

        if cv.group_folds:
            groups = _group_ids(dataset, split)
            splitter = GroupKFold(n_splits=cv.n_splits)
            for train_idx, test_idx in splitter.split(indices, groups=groups):
                train_idx, val_idx = _carve(dataset, cv, split, train_idx)
                fold = DataSplit(
                    train=_subset(dataset, train_idx),
                    val=_subset(dataset, val_idx),
                    test=_subset(dataset, test_idx),
                )
                assert_no_group_leakage(fold)
                folds.append(fold)
        else:
            splitter = KFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.seed)
            for train_idx, test_idx in splitter.split(indices):
                train_idx, val_idx = _carve(dataset, cv, split, train_idx)
                folds.append(
                    DataSplit(
                        train=_subset(dataset, train_idx),
                        val=_subset(dataset, val_idx),
                        test=_subset(dataset, test_idx),
                    )
                )
        return folds


class LosoCVStrategy:
    def generate(
        self, dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig
    ) -> List[DataSplit]:
        folds: List[DataSplit] = []
        groups = _group_ids(dataset, split)
        for g in np.unique(groups):
            test_idx = np.where(groups == g)[0]
            train_idx = np.where(groups != g)[0]
            train_idx, val_idx = _carve(dataset, cv, split, train_idx)
            fold = DataSplit(
                train=_subset(dataset, train_idx),
                val=_subset(dataset, val_idx),
                test=_subset(dataset, test_idx),
            )
            assert_no_group_leakage(fold)
            folds.append(fold)
        return folds


register_cv("kfold")(KFoldCVStrategy)
register_cv("loso")(LosoCVStrategy)
