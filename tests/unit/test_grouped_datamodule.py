import numpy as np

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.configuration.schema import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.configuration.schema import GroupedWindowedPartitions
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import get_windowing_strategy
from hand_gesture_recognition.pipeline.models.neural.datamodule import GroupedEMGDataModule


def _grouped_partitions():
    """Build a GroupedWindowedPartitions via the two_level windowing strategy.

    One user with two trials (a 100-sample fist gesture and a 100-sample rest
    trial) windowed at big_ws=60/big_stride=30 and small_ws=20/small_stride=10
    yields 4 big windows (2 per trial), each with 5 small windows.
    """
    meta = DatasetMetadata("x", "0", 200.0, ["c1", "c2"], ["noGesture", "fist"], "h")
    u = User(
        "s1",
        [
            Trial(
                "t1",
                "s1",
                "fist",
                0,
                100,
                np.arange(100, dtype=np.float32)[:, None].repeat(2, axis=1),
            ),
            Trial("t2", "s1", "noGesture", -1, -1, np.zeros((100, 2), np.float32)),
        ],
    )
    ds = Dataset(meta, [u])
    cfg = WindowingConfig(
        mode="two_level",
        window_size_samples=20,
        stride_samples=10,
        big_window_size_samples=60,
        big_stride_samples=30,
    )
    grouped = get_windowing_strategy("two_level").window(ds, cfg, signal_steps=[])
    return GroupedWindowedPartitions(train=grouped, val=grouped, test=grouped)


def test_grouped_datamodule_shapes():
    partitions = _grouped_partitions()
    feature_specs = [
        {"_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"}
    ]
    n_features = 7 * 2  # 7 time-domain features x 2 channels
    max_small_per_big = max(len(s) for s in partitions.train.small_windows)
    dm = GroupedEMGDataModule(
        windowed=partitions,
        feature_specs=feature_specs,
        n_features=n_features,
        max_small_per_big=max_small_per_big,
        batch_size=2,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # 2 big windows per batch, seq len max_small_per_big, n_features columns
    assert batch["x"].shape == (2, max_small_per_big, n_features)
    assert batch["y_true"].shape == (2,)
    assert batch["y_true"].tolist() == [1, 1]
    assert batch["window_id"][0] == "s1_t1_big0"
    assert batch["trial_id"][0] == "t1"
    assert batch["subject_id"][0] == "s1"
