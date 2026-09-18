from __future__ import annotations

import torch.nn as nn

from hand_gesture_recognition.neural.models.base import ClassifierModule


class LSTMModule(ClassifierModule):
    """LSTM over (time, channels) -> last hidden state -> dense -> logits."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        optimizer=None,
    ):
        super().__init__(optimizer=optimizer)
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, time, channels)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])  # last layer's hidden state: (batch, hidden_size)
