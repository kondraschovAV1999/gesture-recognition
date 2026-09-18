import os

from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.reporting.visualization import plot_raw_signal_by_class


def test_plot_raw_signal_by_class(tmp_path):
    ds = make_synthetic_dataset(
        n_subjects=2,
        n_trials_per_subject=3,
        n_classes=3,
        length=100,
        n_channels=4,
        seed=0,
    )
    paths = plot_raw_signal_by_class(
        ds, str(tmp_path), class_names=ds.metadata.class_names
    )
    assert len(paths) == 3  # classes 0,1,2 all present
    for p in paths:
        assert os.path.exists(p)
    assert os.path.exists(str(tmp_path / "raw_signal_class_0.png"))
