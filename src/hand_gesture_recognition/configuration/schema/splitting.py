from __future__ import annotations

from dataclasses import dataclass

from hand_gesture_recognition.configuration.errors import ConfigValidationError
from hand_gesture_recognition.configuration.schema.dataset import Dataset


@dataclass
class SplitConfig:
    strategy: str = (
        "group_wise"  # random | stratified | group_wise | session_wise | predefined
    )
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    split_on: str = "subject_ids"  # which Dataset field carries group identity

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.0 < self.train_ratio < 1.0):
            raise ConfigValidationError(
                f"train_ratio must be in (0,1), got {self.train_ratio}"
            )
        if abs(total - 1.0) > 1e-6:
            raise ConfigValidationError(f"split ratios must sum to 1.0, got {total}")
        if self.strategy not in (
            "random",
            "stratified",
            "group_wise",
            "session_wise",
            "trial_wise",
            "predefined",
        ):
            raise ConfigValidationError(f"unknown split strategy: {self.strategy}")


@dataclass
class CrossValidationConfig:
    enabled: bool = False
    method: str = "kfold"  # kfold | loso
    n_splits: int = 5  # kfold only
    group_folds: bool = True  # kfold only: subject-wise vs trial-level
    include_val: bool = True
    val_ratio: float = 0.2  # fraction of the training fold carved into val
    seed: int = 42
    report_per_fold: bool = True

    def validate(self) -> None:
        if self.method not in ("kfold", "loso"):
            raise ConfigValidationError(f"unknown cv method: {self.method!r}")
        if self.method == "kfold" and self.n_splits < 2:
            raise ConfigValidationError(
                f"cv.n_splits must be >= 2, got {self.n_splits}"
            )
        if not (0.0 < self.val_ratio < 1.0):
            raise ConfigValidationError(
                f"cv.val_ratio must be in (0,1), got {self.val_ratio}"
            )


@dataclass
class DataSplit:
    train: Dataset
    val: Dataset
    test: Dataset
