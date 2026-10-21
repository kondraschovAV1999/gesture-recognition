from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.data.windowing.two_level import (
    TwoLevelWindowedDataset,
    TwoLevelWindowedPartitions,
)


class TwoLevelEMGDataset(Dataset):
    """One item per big window: a fixed-length sequence of small-window features."""

    def __init__(
        self,
        twd: TwoLevelWindowedDataset,
        feature_specs: List[Any],
        n_features: int,
        max_small_per_big: int,
    ):
        self.y_true = twd.y
        self.window_ids = twd.metadata.window_ids
        self.trial_ids = twd.metadata.trial_ids
        self.subject_ids = twd.metadata.subject_ids
        self.sequences = []
        for smalls in twd.small_windows:
            parts = []
            for spec in feature_specs:
                step = instantiate(spec)
                parts.append(step.transform(smalls))
            feats = np.concatenate(parts, axis=1).astype(
                np.float32
            )  # (n_small, n_features)
            if feats.shape[0] > max_small_per_big:
                feats = feats[:max_small_per_big]
            elif feats.shape[0] < max_small_per_big:
                pad = np.zeros(
                    (max_small_per_big - feats.shape[0], n_features), dtype=np.float32
                )
                feats = np.vstack([feats, pad])
            self.sequences.append(feats)

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


class TwoLevelEMGDataModule(LightningDataModule):
    def __init__(
        self,
        windowed: TwoLevelWindowedPartitions,
        feature_specs: List[Any],
        n_features: int,
        max_small_per_big: int,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.windowed = windowed
        self.feature_specs = feature_specs
        self.n_features = n_features
        self.max_small_per_big = max_small_per_big
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self._train = TwoLevelEMGDataset(
            self.windowed.train,
            self.feature_specs,
            self.n_features,
            self.max_small_per_big,
        )
        self._val = TwoLevelEMGDataset(
            self.windowed.val,
            self.feature_specs,
            self.n_features,
            self.max_small_per_big,
        )
        self._test = TwoLevelEMGDataset(
            self.windowed.test,
            self.feature_specs,
            self.n_features,
            self.max_small_per_big,
        )

    def _loader(self, ds, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
