from __future__ import annotations

from typing import List

import torch.nn as nn

from hand_gesture_recognition.neural.models.base import ClassifierModule


class CNNLSTMModule(ClassifierModule):
    """Composite CNN + LSTM architecture for windowed sEMG.

    The CNN extracts local temporal features per time step while preserving the
    time axis (padding keeps the sequence length), then an LSTM models the
    resulting feature sequence, and a dense head maps the last hidden state to
    logits. This couples a spatial/time-local feature extractor with a
    recurrent temporal model, a common sEMG gesture-recognition design.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_filters: List[int] = (64, 128),
        kernel_size: int = 3,
        hidden_size: int = 64,
        num_layers: int = 1,
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
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, time, channels) -> (batch, channels, time) -> (batch, time, features)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # (batch, hidden_size)
