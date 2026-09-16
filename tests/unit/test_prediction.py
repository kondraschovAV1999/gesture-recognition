import numpy as np

from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.data.windowing.base import WindowMetadata, WindowedDataset


def _windowed(n=10):
    return WindowedDataset(
        X=np.zeros((n, 40, 8), dtype=np.float32),
        y=np.arange(n) % 3,
        metadata=WindowMetadata(
            window_start=np.arange(n),
            window_end=np.arange(n) + 40,
            subject_ids=np.array([f"s{i % 2}" for i in range(n)]),
            trial_ids=np.array([f"t{i}" for i in range(n)]),
        ),
    )


def test_sklearn_adapter_builds_prediction():
    wd = _windowed()
    adapter = SklearnPredictionAdapter()
    pred = adapter.to_prediction(
        windowed=wd,
        y_pred=np.array([0] * 10),
        y_score=np.ones((10, 3)) / 3,
        run_id="r1",
        duration_s=0.1,
    )
    assert pred.backend == "classical"
    assert pred.run_id == "r1"
    assert pred.sample_ids.tolist() == [f"t{i}" for i in range(10)]
    assert pred.y_true.tolist() == (np.arange(10) % 3).tolist()
    assert pred.y_score.shape == (10, 3)
