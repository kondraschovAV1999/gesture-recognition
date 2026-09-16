import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.hashing import hash_files, hash_directory


def _meta() -> DatasetMetadata:
    return DatasetMetadata(
        name="test",
        version="1",
        sampling_rate=200.0,
        channels=["ch1", "ch2"],
        class_names=["a", "b"],
        content_hash="h",
        provenance={},
    )


def test_dataset_n_trials_and_shapes():
    d = Dataset(
        trials=[np.zeros((100, 2)), np.ones((100, 2))],
        labels=np.array([0, 1]),
        subject_ids=np.array(["s1", "s1"]),
        trial_ids=np.array(["t1", "t2"]),
        gesture_start=np.array([0, 0]),
        gesture_end=np.array([100, 100]),
        metadata=_meta(),
    )
    assert d.n_trials == 2
    assert d.trials[0].shape == (100, 2)
    assert d.labels.tolist() == [0, 1]


def test_hash_files_is_order_independent(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    h1 = hash_files([str(a), str(b)])
    h2 = hash_files([str(b), str(a)])
    assert h1 == h2
    assert len(h1) == 64


def test_hash_directory_is_location_independent(tmp_path):
    dir1 = tmp_path / "d1"
    dir2 = tmp_path / "d2"
    dir1.mkdir()
    dir2.mkdir()
    for d in (dir1, dir2):
        (d / "a.txt").write_text("1")
        (d / "b.txt").write_text("2")
    assert hash_directory(str(dir1)) == hash_directory(str(dir2))
