from __future__ import annotations

import logging
import time

import lightning.pytorch as pl
from hydra.utils import instantiate

from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import GroupedWindowedPartitions
from hand_gesture_recognition.pipeline.models.neural.adapter import (
    LightningPredictionAdapter,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import GroupedEMGDataModule
from hand_gesture_recognition.pipeline.evaluation.prediction import PartitionPredictions
from hand_gesture_recognition.pipeline.models.base import TrainingStrategy, register_training
from hand_gesture_recognition.pipeline.feature_extraction.features import FeatureExtractor


@register_training("neural")
class NeuralTrainingStrategy(TrainingStrategy):
    """Neural (PyTorch Lightning) train+prediction.

    Reproduces the old ``_train_and_predict_neural`` /
    ``_train_and_predict_neural_two_level`` behaviour: seed + quiet Lightning
    logging, build the datamodule (flat ``EMGDataModule`` for
    ``WindowedPartitions``, grouped ``GroupedEMGDataModule`` for
    ``GroupedWindowedPartitions``), fit, predict train/val/test with timing, and
    normalize through the Lightning prediction adapter.
    """

    def train_predict(self, windowed, class_names, cfg) -> PartitionPredictions:
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
        pl.seed_everything(cfg.seed, workers=True)

        feature_specs = cfg.feature_extraction.resolve_features(
            cfg.dataset.sampling_rate
        )
        n_classes = len(class_names)

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
            datamodule = GroupedEMGDataModule(
                windowed=windowed,
                feature_specs=feature_specs,
                n_features=n_features,
                max_small_per_big=max_small_per_big,
                batch_size=cfg.neural.datamodule.get("batch_size", 32),
                num_workers=cfg.neural.datamodule.get("num_workers", 0),
            )
            n_channels = n_features
        else:
            n_channels = windowed.train.X.shape[-1]
            dm_partial = instantiate(cfg.neural.datamodule, _partial_=True)
            datamodule = dm_partial(windowed=windowed)

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
