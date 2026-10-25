from __future__ import annotations

from typing import List, Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.configuration.schema import (
    WindowedDataset,
    WindowedPartitions,
)
from hand_gesture_recognition.configuration.schema import (
    GroupedWindowedDataset,
    GroupedWindowedPartitions,
)
from hand_gesture_recognition.pipeline.feature_extraction.features import (
    FeatureExtractor,
)


class EMGWindowDataset(Dataset):
    """Maps a `WindowedDataset` to a torch Dataset whose items carry metadata."""

    def __init__(self, windowed: WindowedDataset):
        self.windowed = windowed

    def __len__(self):
        return len(self.windowed.y)

    def __getitem__(self, idx):
        return {
            "x": torch.as_tensor(self.windowed.X[idx], dtype=torch.float32),
            "y_true": torch.as_tensor(self.windowed.y[idx], dtype=torch.long),
            "window_id": self.windowed.metadata.window_ids[idx],
            "trial_id": self.windowed.metadata.trial_ids[idx],
            "subject_id": self.windowed.metadata.subject_ids[idx],
            "window_start": self.windowed.metadata.window_start[idx],
            "window_end": self.windowed.metadata.window_end[idx],
        }


class EMGDataModule(LightningDataModule):
    """Wraps already-windowed partitions into DataLoaders (no windowing here)."""

    def __init__(
        self,
        windowed: WindowedPartitions,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = False,
    ):
        super().__init__()
        self.windowed = windowed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = EMGWindowDataset(self.windowed.train)
        self._val = EMGWindowDataset(self.windowed.val)
        self._test = EMGWindowDataset(self.windowed.test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()


class GroupedEMGDataset(Dataset):
    """One item per big window: a fixed-length sequence of small-window features.

    Absorbs the old ``TwoLevelEMGDataset``: each big window's small windows are
    transformed by a ``FeatureExtractor`` into a ``(n_small, n_features)``
    matrix, then padded/truncated to a fixed ``max_small_per_big`` sequence
    length (zero-pad the tail / truncate to the head, matching the old
    two-level behaviour).
    """

    def __init__(
        self,
        grouped: GroupedWindowedDataset,
        feature_specs: List,
        max_small_per_big: int,
    ):
        self.y_true = grouped.y
        self.window_ids = grouped.metadata.window_ids
        self.trial_ids = grouped.metadata.trial_ids
        self.subject_ids = grouped.metadata.subject_ids
        self.sequences = FeatureExtractor(feature_specs).transform_grouped(
            grouped.small_windows, max_small_per_big
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "x": torch.as_tensor(self.sequences[idx], dtype=torch.float32),
            "y_true": torch.as_tensor(self.y_true[idx], dtype=torch.long),
            "window_id": self.window_ids[idx],
            "trial_id": self.trial_ids[idx],
            "subject_id": self.subject_ids[idx],
        }


class GroupedEMGDataModule(LightningDataModule):
    """Wraps grouped (two-level) windowing into DataLoaders of feature sequences."""

    def __init__(
        self,
        windowed: GroupedWindowedPartitions,
        feature_specs: List,
        n_features: int,
        max_small_per_big: int,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.windowed = windowed
        self.feature_specs = feature_specs
        # unused: the feature width is now derived inside FeatureExtractor.
        # Kept for backward-compatible construction by NeuralTrainingStrategy.
        self.n_features = n_features
        self.max_small_per_big = max_small_per_big
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = GroupedEMGDataset(
            self.windowed.train,
            self.feature_specs,
            self.max_small_per_big,
        )
        self._val = GroupedEMGDataset(
            self.windowed.val,
            self.feature_specs,
            self.max_small_per_big,
        )
        self._test = GroupedEMGDataset(
            self.windowed.test,
            self.feature_specs,
            self.max_small_per_big,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
