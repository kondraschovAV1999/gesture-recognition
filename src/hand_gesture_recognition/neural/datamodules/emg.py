from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    WindowedPartitions,
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
