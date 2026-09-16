from __future__ import annotations

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule


class ClassifierModule(LightningModule):
    """Shared training/validation/test/predict plumbing for classifiers.

    Subclasses implement `forward(x) -> logits` (shape (batch, n_classes)).
    """

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr

    def _step(self, batch, prefix: str) -> torch.Tensor:
        x, y = batch
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
        x, _ = batch
        return self(x)  # logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
