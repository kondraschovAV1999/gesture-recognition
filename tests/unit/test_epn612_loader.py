import json

from hand_gesture_recognition.pipeline.data_loaders.epn612 import EPN612Loader


def _write_user(root, subdir, user_id, n_samples=3, length=50, testing_labeled=None):
    """Write a user JSON mirroring the real EMG-EPN612 structure.

    `testing_labeled` defaults to True for the trainingJSON cohort (its
    testingSamples are labeled) and False for the testingJSON cohort (its
    testingSamples are unlabeled / held-out).
    """
    if testing_labeled is None:
        testing_labeled = "training" in subdir

    d = root / subdir / f"user{user_id}"
    d.mkdir(parents=True, exist_ok=True)
    gestures = ["noGesture", "fist", "open"]

    def make_samples(n, labeled, with_ground_truth):
        samples = {}
        for i in range(n):
            g = gestures[i % len(gestures)]
            sample = {
                "startPointforGestureExecution": 0,
                "emg": {
                    f"ch{c}": [float(c + j % 5) for j in range(length)]
                    for c in range(1, 9)
                },
            }
            if labeled:
                sample["gestureName"] = g
            if with_ground_truth and labeled and g != "noGesture":
                sample["groundTruthIndex"] = [10, 40]
            samples[f"idx_{i + 1}"] = sample
        return samples

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
        "userInfo": {
            "name": f"user{user_id}",
            "age": 20 + user_id,
            "gender": "man" if user_id % 2 else "woman",
            "occupation": "student",
            "ethnicGroup": "latin",
            "handedness": "right",
            "ArmDamage": "False",
            "distanceFromElbowToMyoInCm": 6,
            "distanceFromElbowToUlnaInCm": 24,
            "armPerimeterInCm": 23,
            "date": "30-Oct-2019",
        },
        "trainingSamples": make_samples(
            n_samples, labeled=True, with_ground_truth=True
        ),
        "testingSamples": make_samples(
            n_samples, labeled=testing_labeled, with_ground_truth=False
        ),
    }
    (d / f"user{user_id}.json").write_text(json.dumps(doc))


def test_epn612_loader_builds_dataset(tmp_path):
    for sub, uid in [("trainingJSON", 1), ("trainingJSON", 2), ("testingJSON", 3)]:
        _write_user(tmp_path, sub, uid)
    loader = EPN612Loader()
    ds = loader.load(str(tmp_path))

    # train cohort: 3 training + 3 (labeled) testing = 6 each; test cohort: 3
    # training (its testingSamples are unlabeled and skipped) = 3.
    assert ds.n_trials == 15
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
    assert ds.trials[0].emg.shape == (50, 8)
    assert set(ds.subject_ids.tolist()) == {"train:user1", "train:user2", "test:user3"}
    # trial ids embed the sample section
    assert any(":trainingSamples:" in t for t in ds.trial_ids)
    assert any(":testingSamples:" in t for t in ds.trial_ids)
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
    # trainingSamples (3): noGesture [-1,-1], fist [10,40], open [10,40]
    # testingSamples (3, labeled, no groundTruthIndex): noGesture [-1,-1],
    #   fist [0,50], open [0,50] (fallback startPoint=0 -> end of recording)
    assert [t.start for t in ds.trials] == [-1, 10, 10, -1, 0, 0]
    assert [t.end for t in ds.trials] == [-1, 40, 40, -1, 50, 50]


def test_epn612_loader_extracts_subject_info(tmp_path):
    _write_user(tmp_path, "trainingJSON", 1)
    _write_user(tmp_path, "testingJSON", 2)
    ds = EPN612Loader().load(str(tmp_path))

    by_id = {u.user_id: u.demographics for u in ds.users}
    assert set(by_id) == {"train:user1", "test:user2"}
    assert by_id["train:user1"]["age"] == 21  # 20 + user_id
    assert by_id["test:user2"]["age"] == 22
    assert by_id["train:user1"]["gender"] in ("man", "woman")
    assert by_id["train:user1"]["handedness"] == "right"
    assert by_id["train:user1"]["arm_damage"] is False


def test_epn612_loader_skips_unlabeled_testing_samples(tmp_path):
    # testingJSON cohort: testingSamples are unlabeled and must be skipped.
    _write_user(tmp_path, "testingJSON", 3)
    ds = EPN612Loader().load(str(tmp_path))
    assert ds.n_trials == 3  # only trainingSamples (3)
    assert all(":trainingSamples:" in t for t in ds.trial_ids)


def test_epn612_loader_max_users_sampling(tmp_path):
    for uid in range(1, 6):
        _write_user(tmp_path, "trainingJSON", uid)
    for uid in range(1, 5):
        _write_user(tmp_path, "testingJSON", uid)
    ds = EPN612Loader(max_train_users=2, max_test_users=2, user_seed=42).load(
        str(tmp_path)
    )
    sids = set(ds.subject_ids.tolist())
    assert len({s for s in sids if s.startswith("train:")}) == 2
    assert len({s for s in sids if s.startswith("test:")}) == 2


def test_epn612_loader_max_users_reproducible(tmp_path):
    for uid in range(1, 8):
        _write_user(tmp_path, "trainingJSON", uid)
    a = EPN612Loader(max_train_users=3, user_seed=7).load(str(tmp_path))
    b = EPN612Loader(max_train_users=3, user_seed=7).load(str(tmp_path))
    assert set(a.subject_ids.tolist()) == set(b.subject_ids.tolist())


def test_epn612_loader_max_users_clamps(tmp_path):
    for uid in range(1, 4):
        _write_user(tmp_path, "trainingJSON", uid)
    ds = EPN612Loader(max_train_users=10, user_seed=42).load(str(tmp_path))
    assert len({s for s in ds.subject_ids.tolist() if s.startswith("train:")}) == 3


def test_epn612_loader_subjects_precedence_over_max_users(tmp_path):
    for uid in range(1, 4):
        _write_user(tmp_path, "trainingJSON", uid)
    ds = EPN612Loader(subjects=["train:user1"], max_train_users=2, user_seed=42).load(
        str(tmp_path)
    )
    assert set(ds.subject_ids.tolist()) == {"train:user1"}


def _write_nested_user(path, cohort="training", user="user1"):
    import os

    d = os.path.join(path, cohort, user)
    os.makedirs(d, exist_ok=True)
    doc = {
        "generalInfo": {"samplingFrequencyInHertz": 200.0},
        "userInfo": {"age": 30},
        "trainingSamples": {
            "s1": {
                "emg": {"ch1": [1, 2, 3], "ch2": [1, 2, 3]},
                "gestureName": "fist",
                "groundTruthIndex": [0, 3],
            },
        },
    }
    with open(os.path.join(d, f"{user}.json"), "w") as f:
        json.dump(doc, f)
    return d


def test_loader_nested(tmp_path):
    _write_nested_user(tmp_path)
    ds = EPN612Loader().load(str(tmp_path))
    assert [u.user_id for u in ds.users] == ["train:user1"]
    assert ds.users[0].demographics["age"] == 30
    t = ds.trials[0]
    assert t.gesture_name == "fist"
    assert t.emg.shape == (3, 2)
    assert (t.start, t.end) == (0, 3)
    assert ds.labels.tolist() == [ds.metadata.class_names.index("fist")]
