from __future__ import annotations

from typing import List

import torch.nn as nn

from hand_gesture_recognition.neural.models.base import ClassifierModule


class CNNModule(ClassifierModule):
    """1-D CNN over the time axis, then global average pooling -> dense head.

    Input ``(batch, time, channels)`` is transposed to ``(batch, channels,
    time)`` so each sEMG channel becomes an input channel to ``nn.Conv1d``.
    A stack of conv + ReLU layers extracts local temporal features; global
    average pooling collapses the time axis into a fixed-size embedding, and a
    dense head maps it to logits.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_filters: List[int] = (64, 128),
        kernel_size: int = 3,
        optimizer=None,
    ):
        super().__init__(optimizer=optimizer)
        layers = []
        in_channels = n_channels
        for filters in conv_filters:
            layers.append(
                nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.ReLU())
            in_channels = filters
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_channels, n_classes)

    def forward(self, x):
        # x: (batch, time, channels) -> (batch, channels, time)
        x = self.conv(x.permute(0, 2, 1))
        x = self.pool(x).squeeze(-1)  # (batch, last_filters)
        return self.head(x)
