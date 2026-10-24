from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.data.splits.base import DataSplit


def _is_dataset_level(step) -> bool:
    """Whether ``step`` operates on a whole :class:`Dataset`.

    Convention: a step whose ``fit`` and ``transform`` both take and return a
    :class:`Dataset` declares the class attribute ``dataset_level = True``
    (e.g. :class:`BaselineRemoval`). Steps without the flag default to
    *per-trial array steps*: ``transform`` takes a single trial's ``emg`` array
    of shape ``(time, channels)`` and returns an array of the same shape
    (e.g. the ``signal.py`` transformers).
    """
    return bool(getattr(step, "dataset_level", False))


def _apply(step, dataset: Dataset) -> Dataset:
    """Apply a single step to a dataset, honoring its level."""
    if _is_dataset_level(step):
        return step.transform(dataset)
    new_trials = [replace(t, emg=step.transform(t.emg)) for t in dataset.trials]
    return dataset.replace_trials(new_trials)


class PreprocessingPipeline:
    """Ordered fit/transform preprocessing (the Template Method seam).

    Steps run in order and are split into two stages:

    - ``pre_steps``: whole-dataset transforms applied once, with no fit
      (e.g. gesture crop). Always dataset-level.
    - ``post_steps``: per-partition transforms. Stateful steps are fit on the
      **train partition only** (fit-on-train), then applied to train/val/test.

    Each step is either *dataset-level* (``dataset_level = True``; ``fit`` and
    ``transform`` take/return a :class:`Dataset`, e.g. ``BaselineRemoval``) or a
    *per-trial array step* (``transform`` maps over each trial's ``emg`` array,
    e.g. the ``signal.py`` transformers). Per-trial array steps are assumed
    deterministic and are not fit.
    """

    def __init__(
        self,
        pre_steps: Optional[List] = None,
        post_steps: Optional[List] = None,
    ):
        self.pre_steps = list(pre_steps or [])
        self.post_steps = list(post_steps or [])

    def fit(self, split: DataSplit) -> "PreprocessingPipeline":
        """Fit stateful (dataset-level) post-steps on the train partition only."""
        for step in self.post_steps:
            if _is_dataset_level(step):
                step.fit(split.train)
        return self

    def transform(self, split: DataSplit) -> DataSplit:
        """Apply fitted post-steps to every partition (fit-on-train only)."""
        return DataSplit(
            train=self._transform_post(split.train),
            val=self._transform_post(split.val),
            test=self._transform_post(split.test),
        )

    def transform_dataset(self, dataset: Dataset) -> Dataset:
        """Apply the pre-steps (whole-dataset transforms) once."""
        out = dataset
        for step in self.pre_steps:
            out = _apply(step, out)
        return out

    def _transform_post(self, dataset: Dataset) -> Dataset:
        out = dataset
        for step in self.post_steps:
            out = _apply(step, out)
        return out
