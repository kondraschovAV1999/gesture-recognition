import numpy as np
import torch

from hand_gesture_recognition.configuration.schema import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
    WindowingConfig,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    get_windowing_strategy,
)
from hand_gesture_recognition.pipeline.models.neural.datamodule import (
    EMGDataset,
    ReshapeToChannelFirstImage,
)

_TIME_SPEC = [
    {
        "_target_": "hand_gesture_recognition.pipeline.feature_extraction.extraction_strategies.time_domain.TimeDomainFeatures"
    }
]


def _grouped():
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
    return get_windowing_strategy("two_level").window(ds, cfg, signal_steps=[])


def test_parallel_conv_dataset_reshapes_to_channels_first():
    grouped = _grouped()
    n_channels = grouped.big.X.shape[-1]  # 2
    max_small = max(len(s) for s in grouped.small_windows)
    ds = EMGDataset.from_grouped(
        grouped,
        _TIME_SPEC,
        max_small,
        transform=ReshapeToChannelFirstImage(n_channels),
    )
    item = ds[0]
    assert isinstance(item["x"], torch.Tensor)
    assert tuple(item["x"].shape) == (n_channels, max_small, 11)  # (C, H, F)
    assert int(item["y_true"].item()) == 1
