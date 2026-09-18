from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

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
        gesture_starts: List[int] = []
        gesture_ends: List[int] = []
        subject_info_rows: Dict[str, Dict[str, Any]] = {}

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
            # User info (demographics) for later analysis.
            ui = doc.get("userInfo", {})
            subject_info_rows[subject_id] = {
                "age": ui.get("age"),
                "gender": ui.get("gender"),
                "occupation": ui.get("occupation"),
                "ethnic_group": ui.get("ethnicGroup"),
                "handedness": ui.get("handedness"),
                "arm_damage": str(ui.get("ArmDamage", "False")).lower() == "true",
                "distance_from_elbow_to_myo_cm": ui.get("distanceFromElbowToMyoInCm"),
                "distance_from_elbow_to_ulna_cm": ui.get("distanceFromElbowToUlnaInCm"),
                "arm_perimeter_cm": ui.get("armPerimeterInCm"),
                "date": ui.get("date"),
            }
            # Load the labeled samples. trainingSamples are always labeled. The
            # train cohort's testingSamples are labeled too (used as the test
            # split); the test cohort's testingSamples are unlabeled (held-out)
            # and therefore skipped.
            sections = ["trainingSamples"]
            if group == "train":
                sections.append("testingSamples")
            for section in sections:
                for key, sample in doc.get(section, {}).items():
                    emg = sample.get("emg")
                    if not emg:
                        continue
                    channels = sorted(
                        emg.keys(), key=lambda c: int(c.replace("ch", ""))
                    )
                    x = np.stack(
                        [np.asarray(emg[c], dtype=np.float32) for c in channels], axis=1
                    )
                    gesture = sample.get("gestureName")
                    if gesture is None or gesture not in label_map:
                        continue
                    # Gesture bounds: groundTruthIndex [start, end] when present,
                    # else startPointforGestureExecution -> end of recording, else
                    # rest trial (-1, -1).
                    if gesture == "noGesture":
                        g_start, g_end = -1, -1
                    else:
                        gti = sample.get("groundTruthIndex")
                        if gti is not None and len(gti) == 2:
                            g_start, g_end = int(gti[0]), int(gti[1])
                        else:
                            g_start = int(
                                sample.get("startPointforGestureExecution", 0)
                            )
                            g_end = x.shape[0]
                    trials.append(x)
                    labels.append(label_map[gesture])
                    subject_ids.append(subject_id)
                    trial_ids.append(f"{subject_id}:{section}:{key}")
                    gesture_starts.append(g_start)
                    gesture_ends.append(g_end)

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
            gesture_start=np.asarray(gesture_starts, dtype=int),
            gesture_end=np.asarray(gesture_ends, dtype=int),
            metadata=DatasetMetadata(
                name="epn612",
                version="2020",
                sampling_rate=sampling_rate,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory_cached(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
            subject_info=pd.DataFrame.from_dict(subject_info_rows, orient="index"),
        )

    def _discover(self, path: str) -> List[str]:
        pattern = os.path.join(path, "**", "user*.json")
        return sorted(glob.glob(pattern, recursive=True))
