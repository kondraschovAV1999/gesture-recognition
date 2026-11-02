from __future__ import annotations

import logging
import time

import lightning.pytorch as pl
from hydra.utils import get_class, instantiate

from hand_gesture_recognition.configuration.schema import PartitionPredictions
from hand_gesture_recognition.pipeline.models.neural.adapter import (
    LightningPredictionAdapter,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import EMGDataModule
from hand_gesture_recognition.pipeline.models.base import (
    TrainingStrategy,
    register_training,
)


@register_training("neural")
class NeuralTrainingStrategy(TrainingStrategy):
    """Neural (PyTorch Lightning) train+prediction.

    Seeds + quiets Lightning logging, asks the model class for its ``InputSpec``
    (input channel/feature count + dataset builder) so it stays model-agnostic,
    builds the ``EMGDataModule``, fits, predicts train/val/test with timing, and
    normalizes through the Lightning prediction adapter.
    """

    def train_predict(self, windowed, class_names, cfg) -> PartitionPredictions:
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
        pl.seed_everything(cfg.seed, workers=True)

        feature_specs = cfg.feature_extraction.resolve_features(
            cfg.dataset.sampling_rate
        )
        n_classes = len(class_names)

        datamodule_cfg = cfg.neural.datamodule
        batch_size = datamodule_cfg.get("batch_size", 32)
        num_workers = datamodule_cfg.get("num_workers", 0)

        # The model class declares how it consumes the data; the strategy only
        # wires the returned dataset builder + channel count into the datamodule
        # and constructor.
        model_cls = get_class(cfg.neural.model["_target_"])
        spec = model_cls.input_spec(windowed, feature_specs)
        datamodule = EMGDataModule(
            windowed,
            spec.dataset_builder,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle_train=datamodule_cfg.get("shuffle", False),
        )

        model_partial = instantiate(
            cfg.neural.model,
            n_channels=spec.n_channels,
            n_classes=n_classes,
            _partial_=True,
            **spec.extra_model_kwargs,
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
