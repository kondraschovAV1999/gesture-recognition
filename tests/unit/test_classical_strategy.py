import numpy as np

from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.configuration.schema import WindowMetadata, WindowedDataset, WindowedPartitions
from hand_gesture_recognition.configuration.schema import GroupedWindowedDataset, GroupedWindowedPartitions
from hand_gesture_recognition.pipeline.models.classical.strategy import ClassicalTrainingStrategy


def _windowed(n, seed, window_size=8, n_channels=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, window_size, n_channels)).astype(np.float32)
    y = np.arange(n) % 3
    meta = WindowMetadata(
        window_start=np.arange(n),
        window_end=np.arange(n) + window_size,
        subject_ids=np.array(["s0"] * n),
        trial_ids=np.array([f"t{i}" for i in range(n)]),
        window_ids=np.array([f"s0_t{i}_window0" for i in range(n)]),
    )
    return WindowedDataset(X=X, y=y, metadata=meta)


def _cfg():
    cfg = ExperimentConfig(run_id="unit-test-run")
    cfg.classical.preprocessing = []
    cfg.classical.model = {
        "_target_": "hand_gesture_recognition.pipeline.models.classical.random_forest.RandomForestEstimator",
        "n_estimators": 10,
        "random_state": 0,
    }
    cfg.feature_extraction.feature_type = "time_domain"
    cfg.dataset.sampling_rate = 200.0
    return cfg


def test_classical_strategy_train_predict():
    windowed = WindowedPartitions(
        train=_windowed(10, seed=0),
        val=_windowed(6, seed=1),
        test=_windowed(6, seed=2),
    )
    predictions = ClassicalTrainingStrategy().train_predict(
        windowed, class_names=["class_0", "class_1", "class_2"], cfg=_cfg()
    )

    for part in ("train", "val", "test"):
        pred = getattr(predictions, part)
        assert pred is not None
        assert pred.backend == "classical"
        assert pred.run_id == "unit-test-run"
        assert len(pred.y_pred) == len(pred.y_true)
        assert pred.y_pred.shape == pred.y_true.shape

    assert len(predictions.train.y_pred) == 10
    assert len(predictions.val.y_pred) == 6
    assert len(predictions.test.y_pred) == 6


def test_classical_strategy_two_level_flattens_to_big_windows():
    def grouped(n, seed):
        big = _windowed(n, seed=seed, window_size=16)
        small_windows = [np.zeros((3, 8, 2), dtype=np.float32) for _ in range(n)]
        return GroupedWindowedDataset(
            big=big,
            small_windows=small_windows,
            big_window_id=big.metadata.window_ids,
            y=big.y,
            metadata=big.metadata,
        )

    windowed = GroupedWindowedPartitions(
        train=grouped(10, seed=0),
        val=grouped(6, seed=1),
        test=grouped(6, seed=2),
    )
    predictions = ClassicalTrainingStrategy().train_predict(
        windowed, class_names=["class_0", "class_1", "class_2"], cfg=_cfg()
    )

    # Trained on the big windows, so prediction counts match big-window counts.
    assert len(predictions.train.y_pred) == 10
    assert len(predictions.val.y_pred) == 6
    assert len(predictions.test.y_pred) == 6
    for part in ("train", "val", "test"):
        pred = getattr(predictions, part)
        assert len(pred.y_pred) == len(pred.y_true)
