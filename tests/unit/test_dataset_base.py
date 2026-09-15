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
        metadata=_meta(),
    )
    assert d.n_trials == 2
    assert d.trials[0].shape == (100, 2)
    assert d.labels.tolist() == [0, 1]


def test_hash_files_is_deterministic(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = hash_files([str(f)])
    h2 = hash_files([str(f)])
    assert h1 == h2
    assert len(h1) == 64


def test_hash_directory_ignores_filenames_but_sorts(tmp_path):
    (tmp_path / "b.txt").write_text("1")
    (tmp_path / "a.txt").write_text("2")
    h = hash_directory(str(tmp_path))
    assert isinstance(h, str) and len(h) == 64
