import json

from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader


def _write_user(root, subdir, user_id, n_samples=3, length=50):
    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "open"]
    samples = {}
    for i in range(n_samples):
        g = gestures[i % len(gestures)]
        sample = {
            "startPointforGestureExecution": 0,
            "gestureName": g,
            "emg": {
                f"ch{c}": [float(c + j % 5) for j in range(length)] for c in range(1, 9)
            },
        }
        if g != "noGesture":
            sample["groundTruthIndex"] = [10, 40]
        samples[f"idx_{i + 1}"] = sample
    doc = {
        "generalInfo": {
            "samplingFrequencyInHertz": 200,
            "myoPredictionLabel": {
                "noGesture": 0,
                "fist": 1,
                "waveIn": 2,
                "waveOut": 3,
                "open": 4,
                "pinch": 5,
            },
        },
        "userInfo": {"name": f"user{user_id}"},
        "trainingSamples": samples,
        "testingSamples": {},
    }
    (d / f"user{user_id}.json").write_text(json.dumps(doc))


def test_epn612_loader_builds_dataset(tmp_path):
    for sub, uid in [("trainingJSON", 1), ("trainingJSON", 2), ("testingJSON", 3)]:
        _write_user(tmp_path, sub, uid)
    loader = EPN612Loader()
    ds = loader.load(str(tmp_path))

    assert ds.n_trials == 9  # 3 users * 3 training samples each
    assert ds.metadata.sampling_rate == 200.0
    assert ds.metadata.channels == [f"ch{i}" for i in range(1, 9)]
    assert set(ds.metadata.class_names) == {
        "noGesture",
        "fist",
        "waveIn",
        "waveOut",
        "open",
        "pinch",
    }
    assert ds.trials[0].shape == (50, 8)
    # subject ids disambiguated by cohort folder, one per trial
    assert set(ds.subject_ids.tolist()) == {"train:user1", "train:user2", "test:user3"}
    assert ds.metadata.content_hash


def test_epn612_loader_subject_filter(tmp_path):
    for sub, uid in [("trainingJSON", 1), ("trainingJSON", 2)]:
        _write_user(tmp_path, sub, uid)
    ds = EPN612Loader(subjects=["train:user1"]).load(str(tmp_path))
    assert set(ds.subject_ids.tolist()) == {"train:user1"}


def test_epn612_loader_subject_filter_is_cohort_aware(tmp_path):
    # same local user number in both cohorts; filter must select only the
    # requested cohort, not both.
    _write_user(tmp_path, "trainingJSON", 1)
    _write_user(tmp_path, "testingJSON", 1)
    ds = EPN612Loader(subjects=["train:user1"]).load(str(tmp_path))
    assert set(ds.subject_ids.tolist()) == {"train:user1"}


def test_epn612_loader_extracts_gesture_bounds(tmp_path):
    _write_user(tmp_path, "trainingJSON", 1)
    ds = EPN612Loader().load(str(tmp_path))
    # idx_1 = noGesture (rest), idx_2/idx_3 = gestures with groundTruthIndex [10, 40]
    assert ds.gesture_start.tolist() == [-1, 10, 10]
    assert ds.gesture_end.tolist() == [-1, 40, 40]
