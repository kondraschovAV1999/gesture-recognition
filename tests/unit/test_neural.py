import numpy as np
import pytest
import torch

from hand_gesture_recognition.configuration.schema import SplitConfig, WindowingConfig
from hand_gesture_recognition.utils.fixtures import make_synthetic_dataset
from hand_gesture_recognition.pipeline.splits.base import split_dataset
from hand_gesture_recognition.configuration.schema import (
    WindowedPartitions,
    WindowMetadata,
    WindowedDataset,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import (
    window_dataset,
)
from hand_gesture_recognition.pipeline.models.neural.adapter import (
    LightningPredictionAdapter,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    EMGDataModule,
    EMGWindowDataset,
)
from hand_gesture_recognition.pipeline.models.neural.dnn import DNNModule
from hand_gesture_recognition.pipeline.models.neural.lstm import LSTMModule
from hand_gesture_recognition.pipeline.models.neural.cnn import CNNModule
from hand_gesture_recognition.pipeline.models.neural.cnn_lstm import CNNLSTMModule


def _windowed_partitions():
    ds = make_synthetic_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        length=100,
        n_channels=8,
        n_classes=3,
        seed=0,
    )
    split = split_dataset(
        ds,
        SplitConfig(
            strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0
        ),
    )
    windowing = WindowingConfig(window_size_samples=40, stride_samples=20)
    return WindowedPartitions(
        train=window_dataset(split.train, windowing),
        val=window_dataset(split.val, windowing),
        test=window_dataset(split.test, windowing),
    )


def test_emg_datamodule_does_not_re_window():
    wp = _windowed_partitions()
    dm = EMGDataModule(windowed=wp, batch_size=8)
    dm.setup()
    X = next(iter(dm.train_dataloader()))
    # batch is a dict carrying metadata
    assert isinstance(X, dict)
    assert set(X.keys()) == {
        "x",
        "y_true",
        "window_id",
        "trial_id",
        "subject_id",
        "window_start",
        "window_end",
    }
    assert X["x"].shape[1:] == (40, 8)


def test_emg_window_dataset_returns_metadata_dict():
    wp = _windowed_partitions()
    ds = EMGWindowDataset(wp.train)
    item = ds[0]
    assert item["x"].shape == (40, 8)
    assert item["y_true"].ndim == 0
    assert isinstance(item["window_id"], str)
    assert isinstance(item["trial_id"], str)


def test_lightning_adapter_reads_metadata_from_predict_output():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    y = torch.tensor([0, 1, 2])
    raw = [
        {
            "logits": logits[:2],
            "y_true": y[:2],
            "window_id": ("w0", "w1"),
            "trial_id": ("t0", "t1"),
            "subject_id": ("s1", "s1"),
        },
        {
            "logits": logits[2:],
            "y_true": y[2:],
            "window_id": ("w2",),
            "trial_id": ("t0",),
            "subject_id": ("s1",),
        },
    ]
    pred = LightningPredictionAdapter().to_prediction(
        raw_predictions=raw, run_id="r", duration_s=0.1
    )
    assert pred.backend == "neural"
    assert pred.sample_ids.tolist() == ["w0", "w1", "w2"]
    assert pred.y_true.tolist() == [0, 1, 2]
    assert pred.y_pred.tolist() == [0, 1, 2]
    assert pred.y_score.shape == (3, 3)


def test_dnn_forward_shape():
    model = DNNModule(n_channels=8, n_classes=3, hidden_dims=(16,))
    logits = model(torch.randn(4, 40, 8))
    assert logits.shape == (4, 3)


def test_lstm_forward_shape():
    model = LSTMModule(n_channels=8, n_classes=3, hidden_size=16, num_layers=1)
    logits = model(torch.randn(4, 40, 8))
    assert logits.shape == (4, 3)


def test_cnn_forward_shape():
    model = CNNModule(n_channels=8, n_classes=3, conv_filters=(16, 32), kernel_size=3)
    logits = model(torch.randn(4, 40, 8))
    assert logits.shape == (4, 3)


def test_cnn_lstm_forward_shape():
    model = CNNLSTMModule(
        n_channels=8,
        n_classes=3,
        conv_filters=(16, 32),
        kernel_size=3,
        hidden_size=16,
        num_layers=1,
    )
    logits = model(torch.randn(4, 40, 8))
    assert logits.shape == (4, 3)


def test_optimizer_is_configurable():
    model = DNNModule(
        n_channels=8,
        n_classes=3,
        hidden_dims=(16,),
        optimizer={"_target_": "torch.optim.SGD", "lr": 0.1, "momentum": 0.9},
    )
    opt = model.configure_optimizers()
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults["lr"] == 0.1
    assert opt.defaults["momentum"] == 0.9


def test_classifier_module_computes_macro_f1():
    model = DNNModule(n_channels=8, n_classes=3, hidden_dims=(16,))
    # F1 metrics are registered per split and compute a valid macro F1.
    assert model.train_f1 is not None
    assert model.val_f1 is not None
    assert model.test_f1 is not None
    f1 = model.val_f1
    f1.update(torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]))  # perfect -> 1.0
    assert float(f1.compute()) == pytest.approx(1.0)
    f1.reset()
    f1.update(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 2]))  # degraded -> < 1.0
    assert 0.0 <= float(f1.compute()) < 1.0
    f1.reset()
