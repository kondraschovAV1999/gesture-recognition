from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from hand_gesture_recognition.pipeline.models.neural.base import (
    ClassifierModule,
    InputSpec,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    EMGDataset,
    ReshapeToChannelFirstSequence,
    compute_n_features,
    max_small_per_big,
)
from hand_gesture_recognition.pipeline.models.neural.parallel_conv import (
    FeatureExtractionModule,
)


class SequenceFolding(nn.Module):
    """Fold the batch and sequence axes so a CNN processes each time step
    independently (the PyTorch analogue of MATLAB's ``sequenceFoldingLayer``).

    Input ``(batch, seq, C, F)`` -> ``(batch * seq, C, 1, F)``: every small
    window becomes its own sample, with a singleton height so the 2-D convolutions
    mix channels and features only (never across time steps).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, c, f = x.shape
        return x.reshape(batch * seq, c, 1, f)


class SequenceUnfolding(nn.Module):
    """Restore the sequence axis (the PyTorch analogue of MATLAB's
    ``sequenceUnfoldingLayer``).

    Input ``(batch * seq, features)`` -> ``(batch, seq, features)`` so an LSTM
    can model the temporal relationships between the per-time-step features.
    """

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_seq, features = x.shape
        return x.reshape(batch_seq // seq_len, seq_len, features)


class CNNFeatureLSTMModule(ClassifierModule):
    """CNN feature extractor + sequence folding/unfolding + LSTM.

    Each small window is a ``(C, F)`` feature "image". The sequence axis is
    folded into the batch so the shared parallel-conv ``FeatureExtractionModule``
    extracts a fixed feature vector per time step; that vector is flattened,
    unfolded back into a sequence, and modelled by an LSTM, with the last hidden
    state mapped to logits by a dense head.

    Input ``(batch, seq, C, F)``; output logits ``(batch, n_classes)``.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        n_features_per_channel: int,
        out_channels: int = 18,
        hidden_size: int = 128,
        num_layers: int = 1,
        optimizer=None,
    ):
        super().__init__(n_classes=n_classes, optimizer=optimizer)
        self.fold = SequenceFolding()
        self.features = FeatureExtractionModule(
            in_channels=n_channels, out_channels=out_channels
        )
        self.unfold = SequenceUnfolding()
        # FeatureExtractionModule concatenates 4 parallel-conv branches.
        lstm_input_size = self.features.out_channels * n_features_per_channel
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, C, F)
        batch, seq, _, _ = x.shape
        folded = self.fold(x)  # (batch*seq, C, 1, F)
        feats = self.features(folded)  # (batch*seq, 4*out, 1, F)
        feats = feats.flatten(1)  # (batch*seq, 4*out*F)
        unfolded = self.unfold(feats, seq)  # (batch, seq, 4*out*F)
        _, (h_n, _) = self.lstm(unfolded)
        return self.head(h_n[-1])  # (batch, n_classes)

    @classmethod
    def input_spec(cls, windowed, feature_specs) -> InputSpec:
        # CNN+LSTM consumes (S, C, F): each small window is a (C, F) image; the
        # sequence axis is folded/unfolded around the CNN.
        n_channels = windowed.train.big.X.shape[-1]
        n_features = compute_n_features(feature_specs, windowed)
        builder = partial(
            EMGDataset.from_grouped,
            feature_specs=feature_specs,
            max_small_per_big=max_small_per_big(windowed),
            transform=ReshapeToChannelFirstSequence(n_channels),
        )
        return InputSpec(
            n_channels=n_channels,
            dataset_builder=builder,
            extra_model_kwargs={"n_features_per_channel": n_features // n_channels},
        )
