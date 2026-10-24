from __future__ import annotations

from typing import List

from hand_gesture_recognition.configuration.schema import (
    CrossValidationConfig,
    SplitConfig,
)
from hand_gesture_recognition.utils.datasets import Dataset
from hand_gesture_recognition.pipeline.splits.base import DataSplit


def generate_folds(
    dataset: Dataset, cv: CrossValidationConfig, split: SplitConfig
) -> List[DataSplit]:
    """Produce the list of DataSplits for a cross-validation run."""
    from hand_gesture_recognition.pipeline.splits.strategy import get_cv_strategy

    cv.validate()
    return get_cv_strategy(cv.method).generate(dataset, cv, split)
