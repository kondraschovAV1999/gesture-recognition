from __future__ import annotations

from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.splits.base import DataSplit
from hand_gesture_recognition.data.windowing.base import WindowedDataset, window_dataset


class EMGDataModule(LightningDataModule):
    """LightningDataModule that windows the split via the SHARED windowing impl.

    The framework split (train/val/test Dataset) is windowed once per partition
    by calling `window_dataset` (no second windowing implementation), then
    wrapped into `TensorDataset`s. Model-agnostic: always yields
    `(batch, time, channels)`.
    """

    def __init__(
        self,
        split: DataSplit,
        windowing: WindowingConfig,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.split = split
        self.windowing = windowing
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train: Optional[TensorDataset] = None
        self._val: Optional[TensorDataset] = None
        self._test: Optional[TensorDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = self._to_dataset(window_dataset(self.split.train, self.windowing))
        self._val = self._to_dataset(window_dataset(self.split.val, self.windowing))
        self._test = self._to_dataset(window_dataset(self.split.test, self.windowing))

    @staticmethod
    def _to_dataset(wd: WindowedDataset) -> TensorDataset:
        X = torch.tensor(wd.X, dtype=torch.float32)
        y = torch.tensor(wd.y, dtype=torch.long)
        return TensorDataset(X, y)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
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
