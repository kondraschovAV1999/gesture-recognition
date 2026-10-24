from __future__ import annotations

import numpy as np

from hand_gesture_recognition.data.splits.base import DataSplit, _subset

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
        # Both strategies split on the group identity given by split_on
        # (subject_ids by default). A true session-wise split needs a
        # session-bearing field; for Phase-1 datasets without one (EPN612 /
        # synthetic) this behaves like group_wise. Group counts guarantee
        # >=1 group per partition whenever the dataset has >=3 groups.
        rng = np.random.default_rng(config.seed)
        groups = (
            dataset.subject_ids
            if config.split_on == "subject_ids"
            else dataset.trial_ids
        )
        unique = np.unique(groups)
        rng.shuffle(unique)
        n = len(unique)
        n_val = max(1, int(round(n * config.val_ratio)))
        n_test = max(1, int(round(n * config.test_ratio)))
        n_train = max(1, n - n_val - n_test)
        if n_train + n_val + n_test > n:  # ratios over-allocate for tiny n
            n_val = max(1, (n - n_train) // 2)
            n_test = n - n_train - n_val
        train_g = unique[:n_train]
        val_g = unique[n_train : n_train + n_val]
        test_g = unique[n_train + n_val : n_train + n_val + n_test]
        train_idx = np.where(np.isin(groups, train_g))[0]
        val_idx = np.where(np.isin(groups, val_g))[0]
        test_idx = np.where(np.isin(groups, test_g))[0]
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
