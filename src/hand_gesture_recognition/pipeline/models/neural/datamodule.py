from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from hand_gesture_recognition.configuration.schema import (
    GroupedWindowedDataset,
    GroupedWindowedPartitions,
    WindowMetadata,
    WindowedDataset,
    WindowedPartitions,
)
from hand_gesture_recognition.pipeline.feature_extraction.features import (
    FeatureExtractor,
)

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class ReshapeToChannelFirstImage:
    """Reshape feature-major ``(..., H, F*C)`` sequences into ``(..., C, H, F)``.

    The handcrafted extractor emits columns ordered ``feat -> channel``; this
    separates them and moves the channel axis in front of the spatial axes so a
    2D conv can treat the small-window axis as height and the feature axis as
    width.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        *batch_dims, height, total_features = x.shape
        f_per_channel = total_features // self.n_channels
        x = x.reshape(*batch_dims, height, f_per_channel, self.n_channels)
        return x.movedim(-1, -3)  # (..., C, height, F)


class EMGDataset(Dataset):
    """Unified dataset for raw windows and grouped handcrafted-feature sequences.

    ``from_windowed`` maps a ``WindowedDataset`` (raw ``(n, time, ch)`` windows).
    ``from_grouped`` maps a ``GroupedWindowedDataset`` by first computing the
    handcrafted small-window features ``(max_small, F*C)``. An optional
    ``transform`` (a pure tensor->tensor callable) is applied once up-front.
    """

    def __init__(
        self,
        x: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        y: Union[np.ndarray, List[int], torch.Tensor],
        metadata: WindowMetadata,
        transform: Optional[TensorTransform] = None,
        include_window_bounds: bool = False,
    ):
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        if transform is not None:
            x_tensor = transform(x_tensor)

        self.x = x_tensor
        self.y_true = torch.as_tensor(np.asarray(y), dtype=torch.long)
        self.metadata = metadata
        self.include_window_bounds = include_window_bounds

    @classmethod
    def from_windowed(
        cls,
        windowed: WindowedDataset,
        transform: Optional[TensorTransform] = None,
    ) -> "EMGDataset":
        return cls(
            x=windowed.X,
            y=windowed.y,
            metadata=windowed.metadata,
            transform=transform,
            include_window_bounds=True,
        )

    @classmethod
    def from_grouped(
        cls,
        grouped: GroupedWindowedDataset,
        feature_specs: List,
        max_small_per_big: int,
        transform: Optional[TensorTransform] = None,
    ) -> "EMGDataset":
        sequences = FeatureExtractor(feature_specs).transform_grouped(
            grouped.small_windows, max_small_per_big
        )
        return cls(
            x=sequences,
            y=grouped.y,
            metadata=cast(WindowMetadata, grouped.metadata),
            transform=transform,
            include_window_bounds=False,
        )

    def __len__(self) -> int:
        return len(self.y_true)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "x": self.x[idx],
            "y_true": self.y_true[idx],
            "window_id": self.metadata.window_ids[idx],
            "trial_id": self.metadata.trial_ids[idx],
            "subject_id": self.metadata.subject_ids[idx],
        }
        if self.include_window_bounds:
            item["window_start"] = self.metadata.window_start[idx]
            item["window_end"] = self.metadata.window_end[idx]
        return item


class EMGDataModule(LightningDataModule):
    """Generic datamodule that wraps any partition set via a dataset factory."""

    def __init__(
        self,
        partitions: Union[WindowedPartitions, GroupedWindowedPartitions],
        dataset_builder: Callable[[Any], Dataset],
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = False,
    ):
        super().__init__()
        self.partitions = partitions
        self.dataset_builder = dataset_builder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self._train: Optional[Dataset] = None
        self._val: Optional[Dataset] = None
        self._test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train = self.dataset_builder(self.partitions.train)
            self._val = self.dataset_builder(self.partitions.val)
        if stage in (None, "test", "predict"):
            self._test = self.dataset_builder(self.partitions.test)

    def _make_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._train, shuffle=self.shuffle_train)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._test, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
