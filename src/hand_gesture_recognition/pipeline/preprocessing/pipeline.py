from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from hand_gesture_recognition.data.datasets.base import Dataset
from hand_gesture_recognition.pipeline.splits.base import DataSplit


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

    The pipeline has two distinct stages that are invoked at **different
    orchestration points** and are **not chained** inside ``transform``:

    - ``pre_steps``: **pre-split** whole-dataset transforms applied once, with
      no fit (e.g. gesture crop). These run on a :class:`Dataset` *before* it is
      split into train/val/test, via :meth:`transform_dataset`.

    - ``post_steps``: **post-split** transforms applied to a :class:`DataSplit`.
      Stateful (dataset-level) steps are fit on the **train partition only**
      (fit-on-train) and then applied to train/val/test, via :meth:`fit` +
      :meth:`transform`.

    :meth:`transform` applies **only** ``post_steps`` — it never runs
    ``pre_steps``. ``fit(split)`` followed by ``transform(split)`` therefore
    does **not** apply ``pre_steps``; the two stages are applied at separate
    points in the orchestration (``transform_dataset`` for the pre-split stage,
    ``fit`` + ``transform`` for the post-split stage).

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
        """Fit stateful (dataset-level) post-steps on the train partition only.

        Only ``post_steps`` participate in fitting. ``pre_steps`` are stateless
        and are applied later, at a separate pre-split point, via
        :meth:`transform_dataset`.
        """
        for step in self.post_steps:
            if _is_dataset_level(step):
                step.fit(split.train)
        return self

    def transform(self, split: DataSplit) -> DataSplit:
        """Apply the fitted ``post_steps`` to every partition (fit-on-train only).

        Applies ``post_steps`` **only**. ``pre_steps`` are *not* applied here —
        they run at a separate pre-split point via :meth:`transform_dataset`.
        """
        return DataSplit(
            train=self._transform_post(split.train),
            val=self._transform_post(split.val),
            test=self._transform_post(split.test),
        )

    def transform_dataset(self, dataset: Dataset) -> Dataset:
        """Apply the ``pre_steps`` (pre-split, whole-dataset transforms) once.

        Applies ``pre_steps`` **only**. ``post_steps`` are *not* applied here —
        they run at a separate post-split point via :meth:`fit` +
        :meth:`transform`.
        """
        out = dataset
        for step in self.pre_steps:
            out = _apply(step, out)
        return out

    def _transform_post(self, dataset: Dataset) -> Dataset:
        """Apply the ``post_steps`` to a single partition's :class:`Dataset`."""
        out = dataset
        for step in self.post_steps:
            out = _apply(step, out)
        return out


class GestureCropStep:
    """Adapt the stateless :func:`~hand_gesture_recognition.pipeline.preprocessing.gesture_crop.crop_to_gesture`
    function to the pipeline step interface.

    Declares ``dataset_level = True`` so the pipeline treats it as a
    whole-dataset transform (``transform`` takes and returns a :class:`Dataset`)
    rather than a per-trial array step. It is a stateless ``pre_steps`` entry: it
    has no ``fit`` and simply delegates to ``crop_to_gesture``.
    """

    dataset_level = True

    def __init__(self, config):
        self.config = config

    def transform(self, dataset: Dataset) -> Dataset:
        from hand_gesture_recognition.pipeline.preprocessing.gesture_crop import crop_to_gesture

        return crop_to_gesture(dataset, self.config)
