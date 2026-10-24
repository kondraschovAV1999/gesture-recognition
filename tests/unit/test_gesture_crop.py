import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
)
from hand_gesture_recognition.data.gesture_crop import crop_to_gesture


def _metadata():
    return DatasetMetadata(
        name="test",
        version="0",
        sampling_rate=200.0,
        channels=["ch1"],
        class_names=["noGesture", "fist"],
        content_hash="x",
    )


def _dataset():
    # gesture trial: length 500, gesture [200, 300]; noGesture trial: length 500
    return Dataset.from_trials(
        [
            Trial(
                "t1",
                "s1",
                "fist",
                200,
                300,
                np.arange(500, dtype=np.float32).reshape(500, 1),
            ),
            Trial(
                "t2",
                "s1",
                "noGesture",
                -1,
                -1,
                np.arange(500, dtype=np.float32).reshape(500, 1),
            ),
        ],
        _metadata(),
    )


def test_crop_gesture_trial_shifts_bounds():
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    ds = crop_to_gesture(_dataset(), cfg)
    # gesture [200,300] + 100 pad -> crop [100,400] -> length 300
    assert ds.trials[0].emg.shape[0] == 300
    assert ds.trials[0].start == 100  # 200 - 100
    assert ds.trials[0].end == 200  # 300 - 100
    # noGesture trial truncated to auto = 2*100 + median(100) = 300
    assert ds.trials[1].emg.shape[0] == 300
    assert ds.trials[1].start == -1


def test_crop_clamps_at_trial_edges():
    # gesture near start: [10, 60], length 200, pad 100
    ds = Dataset.from_trials(
        [
            Trial(
                "t1",
                "s1",
                "fist",
                10,
                60,
                np.arange(200, dtype=np.float32).reshape(200, 1),
            ),
            Trial(
                "t2",
                "s1",
                "noGesture",
                -1,
                -1,
                np.arange(500, dtype=np.float32).reshape(500, 1),
            ),
        ],
        _metadata(),
    )
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    # crop_start = max(0, 10-100)=0; crop_end = min(200, 60+100)=160
    assert out.trials[0].emg.shape[0] == 160
    assert out.trials[0].start == 10  # 10 - 0
    assert out.trials[0].end == 60  # 60 - 0


def test_crop_explicit_no_gesture_length():
    cfg = GestureCropConfig(
        enabled=True, pad_samples=100, no_gesture_max_length_samples=250
    )
    ds = crop_to_gesture(_dataset(), cfg)
    assert ds.trials[1].emg.shape[0] == 250  # explicit value wins over auto


def test_crop_disabled_returns_same_object():
    ds = _dataset()
    cfg = GestureCropConfig(enabled=False, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    assert out is ds


def test_crop_preserves_labels_and_ids():
    ds = _dataset()
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    assert np.array_equal(out.labels, ds.labels)
    assert np.array_equal(out.trial_ids, ds.trial_ids)
    assert np.array_equal(out.subject_ids, ds.subject_ids)
    assert out.metadata is ds.metadata


def test_crop_shifts_bounds():
    from hand_gesture_recognition.data.gesture_crop import crop_to_gesture
    from hand_gesture_recognition.configuration.schema import GestureCropConfig
    import numpy as np
    from hand_gesture_recognition.data.datasets.base import (
        Dataset,
        DatasetMetadata,
        Trial,
        User,
    )

    emg = np.arange(100, dtype=np.float32)[:, None]
    ds = Dataset(
        DatasetMetadata("x", "0", 200.0, ["c"], ["noGesture", "fist"], "h"),
        [User("u1", [Trial("t1", "u1", "fist", 40, 60, emg)])],
    )
    out = crop_to_gesture(ds, GestureCropConfig(enabled=True, pad_samples=10))
    t = out.trials[0]
    assert t.emg.shape[0] == 40  # [30, 70)
    assert (t.start, t.end) == (10, 30)


from dataclasses import replace

from hand_gesture_recognition.configuration.schema import (
    ExperimentConfig,
    TaskConfig,
)
from hand_gesture_recognition.runner.pipeline import Pipeline


def _pipeline(tmp_path, task_name: str, crop_enabled: bool) -> Pipeline:
    cfg = replace(
        ExperimentConfig(),
        task=TaskConfig(name=task_name),
        gesture_crop=GestureCropConfig(enabled=crop_enabled, pad_samples=100),
    )
    return Pipeline(cfg=cfg, logger=None, tracker=None, output_root=str(tmp_path))


def test_pipeline_gesture_crop_respects_flag(tmp_path):
    ds = _dataset()

    # recognition + enabled -> pre-split pipeline crops the gesture trial
    cropped = (
        _pipeline(tmp_path, "recognition", True)
        ._build_preprocessing()
        .transform_dataset(ds)
    )
    assert cropped.trials[0].emg.shape[0] == 300
    assert (cropped.trials[0].start, cropped.trials[0].end) == (100, 200)

    # recognition + disabled -> no crop step, identity transform
    pipeline = _pipeline(tmp_path, "recognition", False)._build_preprocessing()
    assert pipeline.pre_steps == []
    assert pipeline.transform_dataset(ds) is ds

    # classification + enabled -> crop is recognition-only, so no crop step
    pipeline = _pipeline(tmp_path, "classification", True)._build_preprocessing()
    assert pipeline.pre_steps == []
    assert pipeline.transform_dataset(ds) is ds
