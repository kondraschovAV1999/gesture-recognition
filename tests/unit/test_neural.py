import numpy as np
import pytest
import torch

from hand_gesture_recognition.configuration.schema import SplitConfig, WindowingConfig
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import split_dataset
from hand_gesture_recognition.data.windowing.base import WindowMetadata, WindowedDataset
from hand_gesture_recognition.neural.adapters.lightning import (
    LightningPredictionAdapter,
)
from hand_gesture_recognition.neural.datamodules.emg import EMGDataModule
from hand_gesture_recognition.neural.models.dnn import DNNModule
from hand_gesture_recognition.neural.models.lstm import LSTMModule


def _split():
    ds = make_synthetic_dataset(
        n_subjects=4,
        n_trials_per_subject=4,
        length=100,
        n_channels=8,
        n_classes=3,
        seed=0,
    )
    return split_dataset(
        ds,
        SplitConfig(
            strategy="random", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0
        ),
    )


def test_emg_datamodule_windows_via_shared_impl():
    dm = EMGDataModule(
        split=_split(),
        windowing=WindowingConfig(window_size_samples=40, stride_samples=20),
        batch_size=8,
    )
    dm.setup()
    X_batch, y_batch = next(iter(dm.train_dataloader()))
    # windows preserve (time, channels) — the neural model adapts downstream
    assert X_batch.ndim == 3
    assert X_batch.shape[1:] == (40, 8)
    assert X_batch.shape[0] == y_batch.shape[0]


def test_dnn_forward_shape():
    model = DNNModule(n_channels=8, n_classes=3, hidden_dims=(16,))
    logits = model(torch.randn(4, 40, 8))
    assert logits.shape == (4, 3)


def test_lstm_forward_shape():
    model = LSTMModule(n_channels=8, n_classes=3, hidden_size=16, num_layers=1)
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


def test_lightning_adapter_softmax_argmax():
    wd = WindowedDataset(
        X=np.zeros((4, 40, 8), dtype=np.float32),
        y=np.array([0, 1, 2, 1]),
        metadata=WindowMetadata(
            window_start=np.arange(4),
            window_end=np.arange(4) + 40,
            subject_ids=np.array(["s1"] * 4),
            trial_ids=np.array(["t0", "t1", "t2", "t3"]),
        ),
    )
    logits = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 3.0, 0.0],
        ]
    )
    pred = LightningPredictionAdapter().to_prediction(
        windowed=wd, raw_predictions=[logits], run_id="r", duration_s=0.1
    )
    assert pred.backend == "neural"
    assert pred.y_pred.tolist() == [0, 1, 2, 1]
    assert pred.y_score.shape == (4, 3)
    assert pred.y_score[0, 0] == pytest.approx(0.909, abs=1e-2)  # softmax of [3,0,0]
    assert np.array_equal(pred.y_true, wd.y)
    assert pred.sample_ids.tolist() == ["t0", "t1", "t2", "t3"]
