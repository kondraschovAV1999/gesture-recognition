from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from hand_gesture_recognition.data.datasets.base import (
    Dataset,
    DatasetMetadata,
    Trial,
    User,
)
from hand_gesture_recognition.data.hashing import hash_directory_cached
from hand_gesture_recognition.pipeline.data_loaders.base import DatasetLoader

GESTURE_LABELS = {
    "noGesture": 0,
    "fist": 1,
    "waveIn": 2,
    "waveOut": 3,
    "open": 4,
    "pinch": 5,
}


class EPN612Loader(DatasetLoader):
    def __init__(
        self,
        subjects: Optional[Iterable[str]] = None,
        max_train_users: Optional[int] = None,
        max_test_users: Optional[int] = None,
        user_seed: int = 42,
    ):
        # subjects are cohort-qualified ids, e.g. "train:user1" / "test:user3"
        self.subjects = set(subjects) if subjects is not None else None
        self.max_train_users = max_train_users
        self.max_test_users = max_test_users
        self.user_seed = user_seed

    def load(self, path: str) -> Dataset:
        files = self._discover(path)
        if self.subjects is None and (
            self.max_train_users is not None or self.max_test_users is not None
        ):
            self._sample_users(files, path)
        trials_by_user: Dict[str, List[Trial]] = {}
        ordered_subject_ids: List[str] = []
        sampling_rate: Optional[float] = None
        label_map: Optional[Dict[str, int]] = None
        class_names: List[str] = list(GESTURE_LABELS.keys())
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
                    if subject_id not in trials_by_user:
                        trials_by_user[subject_id] = []
                        ordered_subject_ids.append(subject_id)
                    trials_by_user[subject_id].append(
                        Trial(
                            trial_id=f"{subject_id}:{section}:{key}",
                            user_id=subject_id,
                            gesture_name=gesture,
                            start=g_start,
                            end=g_end,
                            emg=x,
                        )
                    )

        if not trials_by_user:
            raise FileNotFoundError(
                f"No usable EPN612 trials found under {path} "
                f"(matched {len(files)} user file(s))"
            )

        users = []
        for sid in ordered_subject_ids:  # preserve first-seen order
            users.append(
                User(
                    user_id=sid,
                    trials=trials_by_user[sid],
                    demographics=subject_info_rows.get(sid),
                )
            )
        return Dataset(
            metadata=DatasetMetadata(
                name="epn612",
                version="2020",
                sampling_rate=sampling_rate,
                channels=[f"ch{i}" for i in range(1, 9)],
                class_names=class_names,
                content_hash=hash_directory_cached(path),
                provenance={"source": "EMG-EPN612", "loader": "EPN612Loader"},
            ),
            users=users,
        )

    def _sample_users(self, files: List[str], path: str) -> None:
        """Restrict `self.subjects` to a seeded-random sample per cohort."""
        rng = np.random.default_rng(self.user_seed)
        train_dirs: set = set()
        test_dirs: set = set()
        for f in files:
            rel = os.path.relpath(f, path)
            parts = rel.split(os.sep)
            cohort_dir = parts[0] if parts else ""
            subject_dir = parts[1] if len(parts) > 1 else ""
            if "training" in cohort_dir.lower():
                train_dirs.add(subject_dir)
            else:
                test_dirs.add(subject_dir)
        train_dirs = sorted(train_dirs)
        test_dirs = sorted(test_dirs)

        def sample(dirs, n):
            if n is None:
                return set(dirs)
            k = min(n, len(dirs))
            return set(
                rng.choice(
                    np.asarray(dirs, dtype=object), size=k, replace=False
                ).tolist()
            )

        allowed = {f"train:{u}" for u in sample(train_dirs, self.max_train_users)}
        allowed |= {f"test:{u}" for u in sample(test_dirs, self.max_test_users)}
        self.subjects = allowed

    def _discover(self, path: str) -> List[str]:
        pattern = os.path.join(path, "**", "user*.json")
        return sorted(glob.glob(pattern, recursive=True))
