from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Optional

import numpy as np

from hand_gesture_recognition.data.datasets.base import Dataset, DatasetMetadata
from hand_gesture_recognition.data.hashing import hash_directory_cached
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
    def __init__(self, subjects: Optional[Iterable[str]] = None):
        # subjects are cohort-qualified ids, e.g. "train:user1" / "test:user3"
        self.subjects = set(subjects) if subjects is not None else None

    def load(self, path: str) -> Dataset:
        files = self._discover(path)
        trials: List[np.ndarray] = []
        labels: List[int] = []
        subject_ids: List[str] = []
        trial_ids: List[str] = []
        sampling_rate: Optional[float] = None
        label_map: Optional[Dict[str, int]] = None
        class_names: List[str] = list(GESTURE_LABELS.keys())

        for f in files:
            # Derive the subject id from the full file path (cohort folder +
            # user subfolder), not from JSON content, so we can filter subjects
            # WITHOUT parsing every file. Layout: <cohort>/<user>/<user>.json.
            rel = os.path.relpath(f, path)
            parts = rel.split(os.sep)
            cohort_dir = parts[0] if parts else ""
            group = "train" if "training" in cohort_dir.lower() else "test"
            subject_dir = parts[1] if len(parts) > 1 else os.path.splitext(parts[-1])[0]
            subject_id = f"{group}:{subject_dir}"
            if self.subjects is not None and subject_id not in self.subjects:
                continue

            with open(f, encoding="utf-8") as fh:
                doc = json.load(fh)
            sampling_rate = float(doc["generalInfo"]["samplingFrequencyInHertz"])
            if label_map is None:
                custom = doc.get("generalInfo", {}).get("myoPredictionLabel")
                label_map = (
                    {str(k): int(v) for k, v in custom.items()}
                    if custom
                    else dict(GESTURE_LABELS)
                )
                class_names = list(label_map.keys())
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
                if gesture not in label_map:
                    continue
                trials.append(x)
                labels.append(label_map[gesture])
                subject_ids.append(subject_id)
                trial_ids.append(f"{subject_id}:{key}")

        if not trials:
            raise FileNotFoundError(
                f"No usable EPN612 trials found under {path} "
                f"(matched {len(files)} user file(s))"
            )

        return Dataset(
            trials=trials,
            labels=np.asarray(labels, dtype=int),
            subject_ids=np.asarray(subject_ids),
            trial_ids=np.asarray(trial_ids),
            metadata=DatasetMetadata(
                name="epn612",
                version="2020",
                sampling_rate=sampling_rate,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory_cached(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
        )

    def _discover(self, path: str) -> List[str]:
        pattern = os.path.join(path, "**", "user*.json")
        return sorted(glob.glob(pattern, recursive=True))
