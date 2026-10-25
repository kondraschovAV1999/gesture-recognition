from __future__ import annotations

from typing import List

import numpy as np

from hand_gesture_recognition.configuration.schema import (
    DataSplit,
    Dataset,
    SplitConfig,
)


def _subset(dataset: Dataset, idx: np.ndarray) -> Dataset:
    chosen = [dataset.trials[int(i)] for i in idx]
    return dataset.replace_trials(chosen)


def split_dataset(dataset: Dataset, config: SplitConfig) -> DataSplit:
    from hand_gesture_recognition.pipeline.splits.strategy import get_split_strategy

    config.validate()
    return get_split_strategy(config.strategy).split(dataset, config)


def assert_no_group_leakage(split: DataSplit, group_field: str = "user_id") -> None:
    def ids(d: Dataset) -> set:
        return {getattr(t, group_field) for t in d.trials}

    tr, va, te = ids(split.train), ids(split.val), ids(split.test)
    if tr & va:
        raise ValueError(f"{group_field} overlap between train and val: {tr & va}")
    if tr & te:
        raise ValueError(f"{group_field} overlap between train and test: {tr & te}")
    if va & te:
        raise ValueError(f"{group_field} overlap between val and test: {va & te}")
