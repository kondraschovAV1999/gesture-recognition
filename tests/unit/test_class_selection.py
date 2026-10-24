import numpy as np
from hand_gesture_recognition.utils.class_selection import (
    build_no_gesture_remap,
    drop_no_gesture,
)
from hand_gesture_recognition.utils.datasets import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)


def _ds():
    meta = DatasetMetadata(
        name="x",
        version="0",
        sampling_rate=200.0,
        channels=["c"],
        class_names=["noGesture", "fist", "open"],
        content_hash="h",
    )
    u = User(
        "u1",
        [
            Trial("t1", "u1", "noGesture", -1, -1, np.zeros((5, 1), np.float32)),
            Trial("t2", "u1", "fist", 0, 5, np.ones((5, 1), np.float32)),
            Trial("t3", "u1", "open", 0, 5, np.full((5, 1), 2.0, np.float32)),
        ],
    )
    return Dataset(meta, [u])


def test_remap_and_drop():
    remap, names = build_no_gesture_remap(_ds())
    assert names == ["fist", "open"]
    assert remap == {1: 0, 2: 1}
    out = drop_no_gesture(_ds(), remap, names)
    assert out.n_trials == 2
    assert out.labels.tolist() == [0, 1]
    assert out.metadata.class_names == ["fist", "open"]
