from __future__ import annotations

from hand_gesture_recognition.configuration.schema import WindowingConfig
from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.windowing.base import (
    WindowedDataset,
    window_dataset,
)
from hand_gesture_recognition.data.windowing.gesture import window_gestures


def build_windowed_dataset(
    dataset: Dataset, config: WindowingConfig, task: str
) -> WindowedDataset:
    """Map a ``task`` name to the matching sample builder.

    ``classification`` segments each gesture into one window (``window_gestures``);
    anything else (``recognition``) slides windows over the continuous signal
    (``window_dataset``). This is the single source of truth shared by the runner
    and the prediction-overlay visualizations, so both re-window the same way the
    training pipeline did.
    """
    if config.mode == "two_level":
        from hand_gesture_recognition.data.windowing.two_level import window_big_windows

        return window_big_windows(dataset, config, [])
    if task == "classification":
        return window_gestures(dataset, config)
    return window_dataset(dataset, config)
