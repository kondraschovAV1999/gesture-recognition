from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from hand_gesture_recognition.configuration.errors import ConfigValidationError
from hand_gesture_recognition.configuration.schema import SplitConfig
from hand_gesture_recognition.data.datasets.base import Dataset


@dataclass
class DataSplit:
    train: Dataset
    val: Dataset
    test: Dataset


def _subset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    subjects_in_subset = [str(s) for s in np.unique(dataset.subject_ids[idx])]
    subject_info = None
    if dataset.subject_info is not None:
        subject_info = dataset.subject_info.loc[
            dataset.subject_info.index.isin(subjects_in_subset)
        ]
    return Dataset(
        trials=[dataset.trials[i] for i in idx],
        labels=dataset.labels[idx],
        subject_ids=dataset.subject_ids[idx],
        trial_ids=dataset.trial_ids[idx],
        gesture_start=dataset.gesture_start[idx],
        gesture_end=dataset.gesture_end[idx],
        metadata=dataset.metadata,
        subject_info=subject_info,
    )


def split_dataset(dataset: Dataset, config: SplitConfig) -> DataSplit:
    config.validate()
    n = dataset.n_trials
    rng = np.random.default_rng(config.seed)

    if config.strategy in ("subject_wise", "session_wise"):
        # Both strategies split on the group identity given by split_on
        # (subject_ids by default). A true session-wise split needs a
        # session-bearing field; for Phase-1 datasets without one (EPN612 /
        # synthetic) this behaves like subject_wise. Group counts guarantee
        # >=1 group per partition whenever the dataset has >=3 groups.
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
    elif config.strategy == "random":
        perm = rng.permutation(n)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        train_idx, val_idx, test_idx = (
            perm[:n_train],
            perm[n_train : n_train + n_val],
            perm[n_train + n_val :],
        )
    elif config.strategy == "stratified":
        from sklearn.model_selection import train_test_split

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
    elif config.strategy == "predefined":
        # EPN612 A1 split: train = trainingJSON.trainingSamples,
        # test = trainingJSON.testingSamples, val = testingJSON.trainingSamples.
        # Subject ids are cohort-qualified ("train:"/"test:"); trial ids embed
        # the sample section (":trainingSamples:" / ":testingSamples:").
        sids = dataset.subject_ids.astype(str)
        tids = dataset.trial_ids.astype(str)
        is_train_cohort = np.char.startswith(sids, "train:")
        is_test_cohort = np.char.startswith(sids, "test:")
        is_training = np.char.find(tids, ":trainingSamples:") >= 0
        is_testing = np.char.find(tids, ":testingSamples:") >= 0
        train_idx = np.where(is_train_cohort & is_training)[0]
        test_idx = np.where(is_train_cohort & is_testing)[0]
        val_idx = np.where(is_test_cohort & is_training)[0]
    else:
        raise ConfigValidationError(f"unknown strategy {config.strategy}")

    return DataSplit(
        train=_subset(dataset, train_idx),
        val=_subset(dataset, val_idx),
        test=_subset(dataset, test_idx),
    )


def assert_no_group_leakage(split: DataSplit, group_field: str = "subject_ids") -> None:
    def ids(d: Dataset) -> set:
        return set(getattr(d, group_field).tolist())

    tr, va, te = ids(split.train), ids(split.val), ids(split.test)
    if tr & va:
        raise ValueError(f"{group_field} overlap between train and val: {tr & va}")
    if tr & te:
        raise ValueError(f"{group_field} overlap between train and test: {tr & te}")
    if va & te:
        raise ValueError(f"{group_field} overlap between val and test: {va & te}")
