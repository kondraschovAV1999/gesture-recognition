from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.hashing import hash_directory
from hand_gesture_recognition.data.loaders.base import DatasetLoader

GESTURE_LABELS = {
    "noGesture": 0,
    "fist": 1,
    "waveIn": 2,
    "waveOut": 3,
    "open": 4,
    "pinch": 5,
}


class EPN612Loader(DatasetLoader):
    def __init__(self, subjects: Optional[Iterable[int]] = None):
        self.subjects = set(subjects) if subjects is not None else None

    def load(self, path: str) -> Dataset:
        files = self._discover(path)
        trials: List[np.ndarray] = []
        labels: List[int] = []
        subject_ids: List[str] = []
        trial_ids: List[str] = []
        sampling_rate: Optional[float] = None
        class_names = list(GESTURE_LABELS.keys())

        for f in files:
            doc = json.load(open(f, encoding="utf-8"))
            user_id = doc["userInfo"]["name"]
            sampling_rate = float(doc["generalInfo"]["samplingFrequencyInHertz"])
            # The dataset ships two cohorts (trainingJSON / testingJSON) that
            # reuse the same local user numbering; disambiguate subject ids by
            # cohort folder so subject-wise splits cannot collide.
            rel = os.path.relpath(f, path)
            group = "train" if "training" in rel.lower() else "test"
            subject_id = f"{group}:{user_id}"
            # Phase 1 loads trainingSamples only; the framework's own split
            # strategy (subject_wise / random) creates train/val/test.
            for key, sample in doc.get("trainingSamples", {}).items():
                emg = sample.get("emg")
                if not emg:
                    continue
                channels = sorted(emg.keys(), key=lambda c: int(c.replace("ch", "")))
                x = np.stack(
                    [np.asarray(emg[c], dtype=np.float32) for c in channels], axis=1
                )
                gesture = sample.get("gestureName", "noGesture")
                if gesture not in GESTURE_LABELS:
                    continue
                trials.append(x)
                labels.append(GESTURE_LABELS[gesture])
                subject_ids.append(subject_id)
                trial_ids.append(f"{subject_id}:{key}")

        if not trials:
            raise FileNotFoundError(f"No EPN612 user files found under {path}")

        return Dataset(
            trials=trials,
            labels=np.asarray(labels, dtype=int),
            subject_ids=np.asarray(subject_ids),
            trial_ids=np.asarray(trial_ids),
            metadata=DatasetMetadata(
                name="epn612",
                version="2020",
                sampling_rate=sampling_rate or 200.0,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
        )

    def _discover(self, path: str) -> List[str]:
        pattern = os.path.join(path, "**", "user*.json")
        files = sorted(glob.glob(pattern, recursive=True))
        if self.subjects is not None:
            files = [f for f in files if self._subject_num(f) in self.subjects]
        return files

    @staticmethod
    def _subject_num(f: str) -> int:
        base = os.path.basename(f)  # e.g. user12.json
        return int("".join(ch for ch in base if ch.isdigit()))
