"""Integration-style tests against the REAL EMG-EPN612 dataset.

These tests are skipped automatically when the dataset is not present at
resources/data/EPN612 (the dataset is git-ignored). They load a small subset
(a single subject) so they stay fast and light on memory.
"""

from __future__ import annotations

import os

import pytest

from hand_gesture_recognition.data.loaders.epn612 import EPN612Loader
from hand_gesture_recognition.paths import resolve_path

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
    assert all(t.ndim == 2 and t.shape[1] == 8 for t in ds.trials)

    # Content hash is a real SHA-256 (64 hex chars), computed over the files.
    assert len(ds.metadata.content_hash) == 64
    assert ds.metadata.content_hash != "synthetic"


def test_load_real_data_preserves_temporal_structure():
    loader = EPN612Loader(subjects=["train:user1"])
    ds = loader.load(DATA_PATH)

    # Trials are continuous ~5s recordings (~1000 samples @ 200 Hz), not
    # pre-windowed: each is far longer than a typical 200ms window.
    lengths = {t.shape[0] for t in ds.trials}
    assert all(length >= 200 for length in lengths)
