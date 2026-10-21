import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.windowing.two_level import (
    TwoLevelWindowedPartitions,
    window_dataset_two_level,
)
from hand_gesture_recognition.neural.datamodules.two_level import TwoLevelEMGDataModule


def _partitions():
    ds = Dataset(
        trials=[np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1)],
        labels=np.array([1]),
        subject_ids=np.array(["s1"]),
        trial_ids=np.array(["t1"]),
        gesture_start=np.array([0]),
        gesture_end=np.array([100]),
        metadata=DatasetMetadata(
            name="x",
            version="0",
            sampling_rate=200.0,
            channels=["c1", "c2"],
            class_names=["rest", "fist"],
            content_hash="h",
        ),
    )
    cfg = WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )
    twd = window_dataset_two_level(ds, cfg, signal_steps=[])
    return TwoLevelWindowedPartitions(train=twd, val=twd, test=twd)


def test_two_level_datamodule_shapes():
    partitions = _partitions()
    feature_specs = [
        {"_target_": "hand_gesture_recognition.features.time_domain.TimeDomainFeatures"}
    ]
    n_features = 7 * 2  # 7 time-domain features x 2 channels
    dm = TwoLevelEMGDataModule(
        windowed=partitions,
        feature_specs=feature_specs,
        n_features=n_features,
        max_small_per_big=6,
        batch_size=2,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # 2 big windows, seq len 6, 14 features
    assert batch["x"].shape == (2, 6, n_features)
    assert batch["y_true"].shape == (2,)
    assert batch["window_id"][0] == "s1_t1_big0"
    assert batch["trial_id"][0] == "t1"
