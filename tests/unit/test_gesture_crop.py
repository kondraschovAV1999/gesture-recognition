import numpy as np

from hand_gesture_recognition.configuration.schema import GestureCropConfig
from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.gesture_crop import crop_to_gesture


def _dataset():
    # gesture trial: length 500, gesture [200, 300]; noGesture trial: length 500
    return Dataset(
        trials=[
            np.arange(500, dtype=np.float32).reshape(500, 1),
            np.arange(500, dtype=np.float32).reshape(500, 1),
        ],
        labels=np.array([1, 0]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([200, -1]),
        gesture_end=np.array([300, -1]),
        metadata=DatasetMetadata(
            name="test",
            version="0",
            sampling_rate=200.0,
            channels=["ch1"],
            class_names=["noGesture", "fist"],
            content_hash="x",
        ),
    )


def test_crop_gesture_trial_shifts_bounds():
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    ds = crop_to_gesture(_dataset(), cfg)
    # gesture [200,300] + 100 pad -> crop [100,400] -> length 300
    assert ds.trials[0].shape[0] == 300
    assert ds.gesture_start[0] == 100  # 200 - 100
    assert ds.gesture_end[0] == 200  # 300 - 100
    # noGesture trial truncated to auto = 2*100 + median(100) = 300
    assert ds.trials[1].shape[0] == 300
    assert ds.gesture_start[1] == -1


def test_crop_clamps_at_trial_edges():
    # gesture near start: [10, 60], length 200, pad 100
    ds = _dataset()
    ds.gesture_start = np.array([10, -1])
    ds.gesture_end = np.array([60, -1])
    ds.trials[0] = np.arange(200, dtype=np.float32).reshape(200, 1)
    cfg = GestureCropConfig(enabled=True, pad_samples=100)
    out = crop_to_gesture(ds, cfg)
    # crop_start = max(0, 10-100)=0; crop_end = min(200, 60+100)=160
    assert out.trials[0].shape[0] == 160
    assert out.gesture_start[0] == 10  # 10 - 0
    assert out.gesture_end[0] == 60  # 60 - 0


def test_crop_explicit_no_gesture_length():
    cfg = GestureCropConfig(
        enabled=True, pad_samples=100, no_gesture_max_length_samples=250
    )
    ds = crop_to_gesture(_dataset(), cfg)
    assert ds.trials[1].shape[0] == 250  # explicit value wins over auto


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


from hand_gesture_recognition.configuration.loader import load_config
from hand_gesture_recognition.runner.run import ExperimentRunner


def test_runner_gesture_crop_respects_flag(tmp_path):
    cfg = load_config(mode="compose", overrides=["dataset.name=synthetic"])
    cfg.tracking.tracking_uri = str(tmp_path / "mlruns")
    runner = ExperimentRunner(cfg)
    ds = _dataset()

    cfg.gesture_crop.enabled = False
    assert runner._gesture_crop(ds) is ds

    cfg.gesture_crop.enabled = True
    cropped = runner._gesture_crop(ds)
    assert cropped.trials[0].shape[0] == 300
