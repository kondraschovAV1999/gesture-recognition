from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch import LightningModule


@dataclass
class InputSpec:
    """How to feed a model: its input channel/feature count and dataset builder.

    ``dataset_builder`` maps a single partition (a ``WindowedDataset`` or
    ``GroupedWindowedDataset``) to a ``torch.utils.data.Dataset``.
    ``extra_model_kwargs`` carries any additional constructor arguments the
    model needs beyond ``n_channels``/``n_classes`` (e.g.
    ``n_features_per_channel``).
    """

    n_channels: int
    dataset_builder: Callable
    extra_model_kwargs: Dict[str, Any] = field(default_factory=dict)


class ClassifierModule(LightningModule):
    """Shared training/validation/test/predict plumbing for classifiers.

    Subclasses implement `forward(x) -> logits` (shape (batch, n_classes)).
    The optimizer is built from `optimizer` config (a `_target_` dict) so the
    optimizer type and its hyperparameters are configurable.
    """

    def __init__(self, n_classes: int, optimizer: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.n_classes = n_classes
        self.optimizer_config = optimizer or {
            "_target_": "torch.optim.Adam",
            "lr": 1e-3,
        }
        # Per-split macro F1 (epoch-level). Direct attributes so the metrics are
        # registered as submodules (and move with the model to the device).
        self.train_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )
        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=n_classes, average="macro"
        )

    def _step(self, batch, prefix: str) -> torch.Tensor:
        x = batch["x"]
        y = batch["y_true"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        f1 = getattr(self, f"{prefix}_f1")
        f1.update(preds, y)
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
        self.log(f"{prefix}_f1", f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["x"]
        logits = self(x)
        return {
            "logits": logits,
            "y_true": batch["y_true"],
            "window_id": batch["window_id"],
            "trial_id": batch["trial_id"],
            "subject_id": batch["subject_id"],
        }

    def configure_optimizers(self):
        from hydra.utils import instantiate

        # Bind the optimizer's config args first (_partial_), then pass the
        # runtime `params` directly (avoids OmegaConf trying to convert the
        # parameter generator).
        opt_partial = instantiate(self.optimizer_config, _partial_=True)
        return opt_partial(self.parameters())

    @classmethod
    def input_spec(cls, windowed, feature_specs) -> InputSpec:
        """Describe how this model consumes data.

        The default handles the two generic sequence cases: raw windows
        (single-level) and handcrafted-feature sequences (two-level). Models
        that consume a reshaped per-channel "image" override this.
        """
        from functools import partial

        from hand_gesture_recognition.configuration.schema import (
            GroupedWindowedPartitions,
        )
        from hand_gesture_recognition.pipeline.models.neural.datamodule import (
            EMGDataset,
            compute_n_features,
            max_small_per_big,
        )

        if isinstance(windowed, GroupedWindowedPartitions):
            builder = partial(
                EMGDataset.from_grouped,
                feature_specs=feature_specs,
                max_small_per_big=max_small_per_big(windowed),
            )
            return InputSpec(
                n_channels=compute_n_features(feature_specs, windowed),
                dataset_builder=builder,
            )
        return InputSpec(
            n_channels=windowed.train.X.shape[-1],
            dataset_builder=EMGDataset.from_windowed,
        )
