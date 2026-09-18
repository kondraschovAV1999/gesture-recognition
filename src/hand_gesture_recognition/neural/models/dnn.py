from __future__ import annotations

from typing import List

import torch.nn as nn

from hand_gesture_recognition.neural.models.base import ClassifierModule


class DNNModule(ClassifierModule):
    """Feed-forward MLP: flattens (time, channels) -> dense layers -> logits.

    `n_channels` is accepted for a uniform constructor with `LSTMModule`; the
    flattened input dimension is inferred by `nn.LazyLinear` on the first
    forward pass.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_dims: List[int] = (128, 64),
        optimizer=None,
    ):
        super().__init__(optimizer=optimizer)
        layers: list = [nn.Flatten(), nn.LazyLinear(hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
