from __future__ import annotations

import logging
import time
from functools import partial

import lightning.pytorch as pl
from hydra.utils import instantiate

from hand_gesture_recognition.configuration.schema import GroupedWindowedPartitions
from hand_gesture_recognition.pipeline.models.neural.adapter import (
    LightningPredictionAdapter,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    EMGDataModule,
    EMGDataset,
    ReshapeToChannelFirstImage,
)
from hand_gesture_recognition.configuration.schema import PartitionPredictions
from hand_gesture_recognition.pipeline.models.base import (
    TrainingStrategy,
    register_training,
)
from hand_gesture_recognition.pipeline.feature_extraction.features import (
    FeatureExtractor,
)


@register_training("neural")
class NeuralTrainingStrategy(TrainingStrategy):
    """Neural (PyTorch Lightning) train+prediction.

    Seeds + quiets Lightning logging, builds a unified ``EMGDataModule`` (a
    per-model ``dataset_builder`` selects raw windows, handcrafted feature
    sequences, or channels-first feature images), fits, predicts
    train/val/test with timing, and normalizes through the Lightning
    prediction adapter.
    """

    def train_predict(self, windowed, class_names, cfg) -> PartitionPredictions:
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
        pl.seed_everything(cfg.seed, workers=True)

        feature_specs = cfg.feature_extraction.resolve_features(
            cfg.dataset.sampling_rate
        )
        n_classes = len(class_names)

        batch_size = cfg.neural.datamodule.get("batch_size", 32)
        num_workers = cfg.neural.datamodule.get("num_workers", 0)

        if isinstance(windowed, GroupedWindowedPartitions):
            n_features = self._compute_n_features(feature_specs, windowed)
            max_small_per_big = max(
                (
                    len(smalls)
                    for part in ("train", "val", "test")
                    for smalls in getattr(windowed, part).small_windows
                ),
                default=1,
            )
            model_target = cfg.neural.model.get("_target_", "")
            if "ParallelConvModule" in model_target:
                # 2D conv consumes (C, H, F): in_channels = EMG channels.
                n_channels = windowed.train.big.X.shape[-1]
                builder = partial(
                    EMGDataset.from_grouped,
                    feature_specs=feature_specs,
                    max_small_per_big=max_small_per_big,
                    transform=ReshapeToChannelFirstImage(n_channels),
                )
            else:
                n_channels = n_features
                builder = partial(
                    EMGDataset.from_grouped,
                    feature_specs=feature_specs,
                    max_small_per_big=max_small_per_big,
                )
            datamodule = EMGDataModule(
                windowed, builder, batch_size=batch_size, num_workers=num_workers
            )
        else:
            n_channels = windowed.train.X.shape[-1]
            datamodule = EMGDataModule(
                windowed,
                EMGDataset.from_windowed,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=cfg.neural.datamodule.get("shuffle", False),
            )

        model_partial = instantiate(
            cfg.neural.model,
            n_channels=n_channels,
            n_classes=n_classes,
            _partial_=True,
        )
        model = model_partial(optimizer=cfg.neural.optimizer)
        callbacks = None
        if cfg.neural.early_stopping:
            callbacks = [instantiate(cfg.neural.early_stopping)]
        trainer_partial = instantiate(cfg.neural.trainer, _partial_=True)
        trainer = trainer_partial(callbacks=callbacks)

        trainer.fit(model, datamodule=datamodule)

        adapter = LightningPredictionAdapter()
        predictions = {}
        for part in ("train", "val", "test"):
            t0 = time.perf_counter()
            raw = trainer.predict(
                model, dataloaders=getattr(datamodule, f"{part}_dataloader")()
            )
            duration = time.perf_counter() - t0
            predictions[part] = adapter.to_prediction(
                raw_predictions=raw,
                run_id=cfg.run_id,
                duration_s=duration,
            )
        return PartitionPredictions(**predictions)

    @staticmethod
    def _compute_n_features(feature_specs, windowed) -> int:
        features = FeatureExtractor(feature_specs)
        for part in ("train", "val", "test"):
            for smalls in getattr(windowed, part).small_windows:
                if smalls.shape[0] > 0:
                    return int(features.transform(smalls[:1]).shape[1])
        return 0
