import torch

from hand_gesture_recognition.pipeline.models.neural.cnn_feature_lstm import (
    CNNFeatureLSTMModule,
    SequenceFolding,
    SequenceUnfolding,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    ReshapeToChannelFirstSequence,
)


def test_sequence_folding_shape():
    fold = SequenceFolding()
    x = torch.randn(4, 7, 3, 5)  # (batch, seq, C, F)
    out = fold(x)
    assert out.shape == (4 * 7, 3, 1, 5)


def test_sequence_unfolding_roundtrip():
    fold = SequenceFolding()
    unfold = SequenceUnfolding()
    x = torch.randn(4, 7, 3, 5)
    folded = fold(x)  # (28, 3, 1, 5)
    flat = folded.flatten(1)  # (28, 15)
    unfolded = unfold(flat, seq_len=7)
    assert unfolded.shape == (4, 7, 15)


def test_reshape_to_channel_first_sequence():
    transform = ReshapeToChannelFirstSequence(n_channels=3)
    # (2 batches, 7 small windows, 5 features x 3 channels = 15 columns)
    x = torch.arange(2 * 7 * 15, dtype=torch.float32).reshape(2, 7, 15)
    out = transform(x)
    assert out.shape == (2, 7, 3, 5)


def test_cnn_feature_lstm_forward_shape():
    model = CNNFeatureLSTMModule(
        n_channels=3,
        n_classes=4,
        n_features_per_channel=5,
        out_channels=4,
        hidden_size=128,
        num_layers=1,
    )
    x = torch.randn(4, 7, 3, 5)  # (batch, seq, C, F)
    logits = model(x)
    assert logits.shape == (4, 4)


def test_cnn_feature_lstm_has_f1_metric():
    model = CNNFeatureLSTMModule(
        n_channels=3, n_classes=4, n_features_per_channel=5, out_channels=4
    )
    assert model.val_f1 is not None
