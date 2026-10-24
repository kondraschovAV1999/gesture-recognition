from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule


class ClassifierModule(LightningModule):
    """Shared training/validation/test/predict plumbing for classifiers.

    Subclasses implement `forward(x) -> logits` (shape (batch, n_classes)).
    The optimizer is built from `optimizer` config (a `_target_` dict) so the
    optimizer type and its hyperparameters are configurable.
    """

    def __init__(self, optimizer: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.optimizer_config = optimizer or {
            "_target_": "torch.optim.Adam",
            "lr": 1e-3,
        }

    def _step(self, batch, prefix: str) -> torch.Tensor:
        x = batch["x"]
        y = batch["y_true"]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True)
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
