"""Integration-style tests against the REAL EMG-EPN612 dataset.

These tests are skipped automatically when the dataset is not present at
resources/data/EPN612 (the dataset is git-ignored). They load a small subset
(a single subject) so they stay fast and light on memory.
"""

from __future__ import annotations

import os

import pytest

from hand_gesture_recognition.pipeline.data_loaders.epn612 import EPN612Loader
from hand_gesture_recognition.utils.paths import resolve_path

DATA_PATH = resolve_path("resources/data/EPN612")
HAS_REAL_DATA = os.path.isdir(DATA_PATH)

pytestmark = pytest.mark.skipif(
    not HAS_REAL_DATA,
    reason="EMG-EPN612 dataset not present under resources/data/EPN612",
)


def test_load_real_data_single_subject():
    loader = EPN612Loader(subjects=["train:user1"])
    ds = loader.load(DATA_PATH)

    # One subject's worth of training samples (150 gestures).
    assert ds.n_trials > 0
    assert set(ds.subject_ids.tolist()) == {"train:user1"}

    # Dataset-level metadata matches the Myo Armband recording.
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

    # Every trial is 2D (time_steps, 8 channels), variable length allowed.
    assert all(t.emg.ndim == 2 and t.emg.shape[1] == 8 for t in ds.trials)

    # Content hash is a real SHA-256 (64 hex chars), computed over the files.
    assert len(ds.metadata.content_hash) == 64
    assert ds.metadata.content_hash != "synthetic"


def test_load_real_data_preserves_temporal_structure():
    loader = EPN612Loader(subjects=["train:user1"])
    ds = loader.load(DATA_PATH)

    # Trials are continuous ~5s recordings (~1000 samples @ 200 Hz), not
    # pre-windowed: each is far longer than a typical 200ms window.
    lengths = {t.emg.shape[0] for t in ds.trials}
    assert all(length >= 200 for length in lengths)


def test_real_data_per_window_labels_match_bounds():
    from hand_gesture_recognition.configuration.schema import WindowingConfig
    from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import (
        window_dataset,
    )

    ds = EPN612Loader(subjects=["train:user1"]).load(DATA_PATH)
    wd = window_dataset(ds, WindowingConfig(window_size_samples=40, stride_samples=10))

    info = {}
    for t, label in zip(ds.trials, ds.labels):
        info[str(t.trial_id)] = (int(label), int(t.start), int(t.end))

    assert wd.X.shape[0] > 0
    for i in range(wd.X.shape[0]):
        label, g_start, g_end = info[str(wd.metadata.trial_ids[i])]
        s = int(wd.metadata.window_start[i])
        e = int(wd.metadata.window_end[i])
        if g_start < 0:  # rest trial
            expected = 0
        elif e <= g_start or s >= g_end:  # fully outside the gesture segment
            expected = 0
        else:  # overlaps the gesture segment
            expected = label
        assert wd.y[i] == expected
