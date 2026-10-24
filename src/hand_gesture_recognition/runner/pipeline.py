"""Template Method ``Pipeline`` orchestrating a full experiment run.

Replaces the god-orchestration in ``ExperimentRunner.run()`` (Phase 5.1). The
runner is reduced to a thin facade that owns tracker lifecycle (start/autolog/
finish), while this class owns the ordered phase sequence

    load -> analytics -> preprocess -> split/CV -> window -> train/predict
    -> evaluate -> track/report

Cross-cutting concerns that vary by configuration (split strategy, CV method,
windowing mode, training backend) are delegated to strategies resolved once at
construction, so the Pipeline body carries no task/backend conditionals.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
from hydra.utils import instantiate

from hand_gesture_recognition.configuration.loader import save_resolved_config
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.pipeline.splits.base import (
    DataSplit,
    assert_no_group_leakage,
)
from hand_gesture_recognition.pipeline.splits.strategy import (
    get_cv_strategy,
    get_split_strategy,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.base import (
    WindowedPartitions,
)
from hand_gesture_recognition.pipeline.feature_extraction.windowing.strategy import (
    GroupedWindowedPartitions,
    get_windowing_strategy,
)
from hand_gesture_recognition.pipeline.evaluation.aggregate import aggregate_folds
from hand_gesture_recognition.pipeline.evaluation.base import EvaluationResult, evaluate
from hand_gesture_recognition.paths import resolve_path
from hand_gesture_recognition.pipeline.preprocessing.pipeline import (
    GestureCropStep,
    PreprocessingPipeline,
)
from hand_gesture_recognition.reporting.reporters import (
    HtmlReporter,
    JsonReporter,
    render_report,
)
from hand_gesture_recognition.tracing.stage import stage

# Import the concrete training strategies so their registry decorators run
# before ``get_training_strategy`` is called in ``Pipeline.__init__``.
from hand_gesture_recognition.pipeline.models.classical.strategy import (  # noqa: F401
    ClassicalTrainingStrategy,
)
from hand_gesture_recognition.pipeline.models.neural.strategy import (  # noqa: F401
    NeuralTrainingStrategy,
)
from hand_gesture_recognition.pipeline.models.base import get_training_strategy

__all__ = ["Pipeline", "PipelineBuilder"]


class PipelineBuilder:
    """Builds a :class:`Pipeline` from the experiment's dependencies.

    Strategy resolution (split, CV, windowing, training) and preprocessing
    construction already live in ``Pipeline.__init__`` (Task 5.1); this builder
    is a thin wrapper so the Phase 5.3 facade can construct a Pipeline without
    coupling to that constructor directly.
    """

    def __init__(self, cfg, logger, tracker, output_root):
        self.cfg = cfg
        self.logger = logger
        self.tracker = tracker
        self.output_root = output_root

    def build(self) -> Pipeline:
        return Pipeline(
            cfg=self.cfg,
            logger=self.logger,
            tracker=self.tracker,
            output_root=self.output_root,
        )


class Pipeline:
    """Template Method orchestrating a full experiment run.

    Strategies (split, CV, windowing, training) are resolved once at
    construction from the resolved config; the preprocessing pipeline is built
    here too. ``run()`` executes the ordered phase sequence. Phase methods are
    private and overridable, so subclasses can specialize a single phase
    without touching the orchestration.
    """

    def __init__(self, cfg, logger, tracker, output_root):
        self.cfg = cfg
        self.logger = logger
        self.tracker = tracker
        self.output_root = output_root

        # Resolve strategies from the resolved config. Windowing mode / task and
        # backend select the concrete strategy, so the Pipeline body below
        # carries no task/backend conditionals.
        self._split_strategy = get_split_strategy(cfg.split.strategy)
        self._cv_strategy = get_cv_strategy(cfg.cv.method)
        windowing_name = (
            "two_level" if cfg.windowing.mode == "two_level" else cfg.task.name
        )
        self._windowing_strategy = get_windowing_strategy(windowing_name)
        self._training_strategy = get_training_strategy(cfg.backend)
        self._preprocessing = self._build_preprocessing()

    def _build_preprocessing(self) -> PreprocessingPipeline:
        """Assemble the two-stage preprocessing pipeline from config.

        ``pre_steps`` (pre-split, whole-dataset, no fit): gesture crop for the
        recognition task. ``post_steps`` (post-split, fit-on-train): baseline
        removal followed by the configured per-trial signal transformers — the
        latter are skipped for ``two_level`` windowing, where they are applied
        per big window inside the windowing strategy (Phase 4).
        """
        cfg = self.cfg
        pre_steps = []
        if cfg.task.name == "recognition" and cfg.gesture_crop.enabled:
            pre_steps.append(GestureCropStep(cfg.gesture_crop))

        post_steps = []
        if cfg.baseline.enabled:
            from hand_gesture_recognition.pipeline.preprocessing.baseline import (
                BaselineRemoval,
            )

            post_steps.append(BaselineRemoval(mode=cfg.baseline.mode))
        if cfg.windowing.mode != "two_level":
            post_steps.extend(instantiate(s) for s in cfg.signal_processing)

        return PreprocessingPipeline(pre_steps=pre_steps, post_steps=post_steps)

    def run(self) -> EvaluationResult:
        """The template method: execute the ordered phase sequence.

        Reproduces ``ExperimentRunner.run()`` (minus tracker start/autolog/finish,
        which the facade owns). Returns the evaluation result for the test set
        (single split) or the cross-fold aggregate (CV).
        """
        cfg = self.cfg

        with stage(self.logger, "load_dataset"):
            raw_dataset = self._load()
        with stage(self.logger, "analytics"):
            self._analytics(raw_dataset)
        with stage(self.logger, "gesture_crop"):
            cropped = self._preprocess_dataset(raw_dataset)

        # Global class set: no-gesture remap computed once from the FULL
        # dataset so every fold/partition shares the same mapping.
        class_names, remap = self._class_setup(cropped)
        dataset_name = cropped.metadata.name

        # Visualization-only baseline (fit on the FULL cropped dataset, a
        # diagnostic; training baselines are fit per partition in
        # _run_partition). Fitted only when visualization needs it.
        viz_baseline = None
        if cfg.baseline.enabled and cfg.visualization.enabled:
            from hand_gesture_recognition.pipeline.preprocessing.baseline import (
                BaselineRemoval,
            )

            viz_baseline = BaselineRemoval(mode=cfg.baseline.mode).fit(cropped)

        if cfg.cv.enabled:
            with stage(self.logger, "split"):
                folds = self._cv_strategy.generate(cropped, cfg.cv, cfg.split)
            fold_test_results = []
            for i, fold in enumerate(folds):
                with stage(self.logger, f"fold_{i}"):
                    _, fold_results, _ = self._run_partition(
                        fold, class_names, remap, dataset_name
                    )
                if fold_results["test"] is not None:
                    fold_test_results.append(fold_results["test"])
            result = aggregate_folds(fold_test_results, cfg.run_id, dataset_name)
            self.logger.info(f"CV metrics (mean): {result.metrics}")
            split, predictions = None, None
            partition_results = None
        else:
            with stage(self.logger, "split"):
                split = self._split(cropped)
            predictions, partition_results, split = self._run_partition(
                split, class_names, remap, dataset_name
            )
            result = partition_results["test"]
            fold_test_results = None

        # processed dataset for visualization (baseline first, then signal).
        processed_dataset = (
            viz_baseline.transform(cropped) if viz_baseline is not None else cropped
        )
        for step_cfg in cfg.signal_processing:
            step = instantiate(step_cfg)
            processed_dataset = processed_dataset.replace_trials(
                [
                    replace(t, emg=step.transform(t.emg))
                    for t in processed_dataset.trials
                ]
            )

        with stage(self.logger, "track_and_report"):
            self._track_and_report(
                cropped=cropped,
                processed_dataset=processed_dataset,
                split=split,
                predictions=predictions,
                class_names=class_names,
                viz_baseline=viz_baseline,
                result=result,
                partition_results=partition_results,
                fold_test_results=fold_test_results,
            )

        return result

    def _load(self):
        """Load the raw dataset (synthetic fixture or configured loader)."""
        if self.cfg.dataset.name == "synthetic":
            return make_synthetic_dataset(seed=self.cfg.seed)
        loader = instantiate(self.cfg.dataset.loader)
        # Resolve the dataset path to a full path anchored at the project root
        # (the config default is a relative path, which only works when the
        # process runs from the repo root).
        return loader.load(resolve_path(self.cfg.dataset.path))

    def _analytics(self, dataset) -> None:
        """Write dataset summary + optional QC artifacts."""
        if not self.cfg.analytics.enabled:
            return
        from hand_gesture_recognition.analytics.stats import dataset_summary

        d = os.path.join(self.output_root, "analytics", dataset.metadata.name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "dataset_summary.json"), "w") as f:
            json.dump(dataset_summary(dataset), f, indent=2)
        if self.cfg.analytics.run_qc:
            from hand_gesture_recognition.analytics.qc import run_quality_checks
            from hand_gesture_recognition.analytics.plots import plot_class_distribution

            with open(os.path.join(d, "quality_checks.json"), "w") as f:
                json.dump(run_quality_checks(dataset), f, indent=2)
            plot_class_distribution(dataset, os.path.join(d, "class_distribution.png"))

    def _preprocess_dataset(self, dataset):
        """Apply the pre-split ``pre_steps`` (gesture crop) to the whole dataset."""
        return self._preprocessing.transform_dataset(dataset)

    def _class_setup(self, cropped):
        """Return ``(class_names, remap)`` after optional no-gesture remapping.

        The remap is computed once from the FULL dataset so every fold/partition
        shares the same label mapping.
        """
        class_names = cropped.metadata.class_names
        remap = None
        if self._should_drop_no_gesture():
            from hand_gesture_recognition.data.class_selection import (
                build_no_gesture_remap,
            )

            remap, class_names = build_no_gesture_remap(cropped)
        return class_names, remap

    def _should_drop_no_gesture(self) -> bool:
        return (not self.cfg.dataset.include_no_gesture) or (
            self.cfg.task.name == "classification"
        )

    def _split(self, dataset):
        """Split the dataset into train/val/test (non-CV path)."""
        split = self._split_strategy.split(dataset, self.cfg.split)
        if self.cfg.split.strategy in ("group_wise", "session_wise", "predefined"):
            assert_no_group_leakage(split)
        return split

    def _run_partition(self, split, class_names, remap, dataset_name):
        """Run preprocessing -> class selection -> window -> train -> evaluate
        on a single DataSplit.

        Returns ``(predictions, {part: result-or-None}, transformed_split)``
        where ``transformed_split`` is the post-class-selection /
        post-preprocessing split used for windowing and prediction (so the
        prediction-overlay visualizations align with predictions).
        """
        # Baseline removal + signal transformers via the fitted pipeline
        # (fit-on-train: the stateful baseline step is fit on `split.train`
        # only, then all partitions are transformed). For two_level windowing
        # the signal steps are excluded from the pipeline and applied per big
        # window inside the windowing strategy instead (Phase 4).
        split = self._preprocessing.fit(split).transform(split)
        if self._should_drop_no_gesture():
            from hand_gesture_recognition.data.class_selection import drop_no_gesture

            split = DataSplit(
                train=drop_no_gesture(split.train, remap, class_names),
                val=drop_no_gesture(split.val, remap, class_names),
                test=drop_no_gesture(split.test, remap, class_names),
            )
        with stage(self.logger, "windowing"):
            windowed = self._window(split)
        with stage(self.logger, "train"):
            predictions = self._training_strategy.train_predict(
                windowed, class_names, self.cfg
            )
        results = {}
        for part in ("train", "val", "test"):
            pred = getattr(predictions, part)
            if pred is None or np.asarray(pred.y_true).size == 0:
                results[part] = None
                continue
            results[part] = evaluate(
                pred,
                self.cfg.evaluation,
                dataset_id=dataset_name,
                split_info={"strategy": self.cfg.split.strategy},
                class_names=class_names,
            )
        if self.cfg.evaluation.breakdown_by:
            self._log_grouped(predictions, class_names, dataset_name)
        return predictions, results, split

    def _log_grouped(self, predictions, class_names, dataset_name) -> None:
        """Log per-group metrics for each configured ``evaluation.breakdown_by`` key.

        For the test partition (and val when present), each group's metrics are
        logged under ``{metric}_{group}`` (e.g. ``accuracy_s01``) via the tracker.
        Non-breaking: this only logs; the returned ``result`` is unchanged.
        """
        from hand_gesture_recognition.pipeline.evaluation.grouped import (
            evaluate_grouped,
        )

        for by in self.cfg.evaluation.breakdown_by:
            for part in ("val", "test"):
                pred = getattr(predictions, part, None)
                if pred is None or np.asarray(pred.y_true).size == 0:
                    continue
                grouped = evaluate_grouped(
                    pred,
                    by,
                    self.cfg.evaluation,
                    dataset_id=dataset_name,
                    class_names=class_names,
                )
                for group, g_result in grouped.items():
                    self.tracker.log_metrics(
                        {
                            f"{metric}_{group}": value
                            for metric, value in g_result.metrics.items()
                        }
                    )

    def _window(self, split):
        """Window each partition via the resolved windowing strategy.

        Single-level windowing returns ``WindowedPartitions``; two-level
        windowing returns ``GroupedWindowedPartitions`` (the strategy's
        ``window`` produces a ``GroupedWindowedDataset`` per partition). For
        two-level, the signal-processing steps are passed through so they run
        per big window inside the strategy.
        """
        two_level = self.cfg.windowing.mode == "two_level"
        signal_steps = (
            [instantiate(s) for s in self.cfg.signal_processing] if two_level else None
        )

        def window_partition(dataset):
            return self._windowing_strategy.window(
                dataset, self.cfg.windowing, signal_steps
            )

        if two_level:
            windowed = GroupedWindowedPartitions(
                train=window_partition(split.train),
                val=window_partition(split.val),
                test=window_partition(split.test),
            )
            for part in ("train", "val", "test"):
                if getattr(windowed, part).y.shape[0] == 0:
                    self.logger.warning(
                        f"partition '{part}' has 0 big windows after two-level windowing"
                    )
            return windowed

        windowed = WindowedPartitions(
            train=window_partition(split.train),
            val=window_partition(split.val),
            test=window_partition(split.test),
        )
        for part in ("train", "val", "test"):
            if getattr(windowed, part).X.shape[0] == 0:
                self.logger.warning(f"partition '{part}' has 0 windows after windowing")
        return windowed

    def _track_and_report(
        self,
        cropped,
        processed_dataset,
        split,
        predictions,
        class_names,
        viz_baseline,
        result,
        partition_results,
        fold_test_results,
    ) -> None:
        """Log dataset/params, report results, visualize, and persist artifacts."""
        self.tracker.log_dataset(cropped)
        self.tracker.log_params({"seed": self.cfg.seed, "backend": self.cfg.backend})
        if self.cfg.cv.enabled:
            self._report_cv(result, fold_test_results, class_names)
            run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
            self._write_cv_summary(fold_test_results, run_dir)
        else:
            for part, r in partition_results.items():
                if r is None:
                    continue
                self.tracker.log_results(
                    r, partition=(None if part == "test" else part)
                )
                self.tracker.log_confusion_matrix(r, class_names, partition=part)
        self._visualize(
            cropped,
            processed_dataset,
            split,
            predictions,
            class_names,
            viz_baseline,
            is_cv=self.cfg.cv.enabled,
        )
        self._persist_artifacts(cropped, result)

    def _report_cv(self, result, fold_results, class_names):
        self.tracker.log_metrics(result.metrics)
        if self.cfg.cv.report_per_fold:
            for i, r in enumerate(fold_results):
                self.tracker.log_results(r, partition=f"fold{i}")
                self.tracker.log_confusion_matrix(
                    r, class_names, partition=f"fold{i}_test"
                )

    def _write_cv_summary(self, fold_results, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        summary = {
            "n_folds": len(fold_results),
            "folds": [
                {"fold": i, "metrics": r.metrics} for i, r in enumerate(fold_results)
            ],
        }
        path = os.path.join(run_dir, "cv_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.tracker.log_artifact(path)

    def _visualize(
        self,
        raw_dataset,
        processed_dataset,
        split,
        predictions,
        class_names,
        viz_baseline=None,
        is_cv=False,
    ) -> None:
        vis = self.cfg.visualization
        if not vis.enabled:
            return
        from hand_gesture_recognition.reporting.visualization import (
            compute_stages,
            plot_fft_spectrum,
            plot_feature_over_windows,
            plot_predictions_on_signal,
            plot_raw_signal_by_class,
            plot_signal_data_flow,
            plot_signal_processing_comparison,
            plot_stage_progression,
            resolve_selection,
        )

        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)

        n_channels = (
            len(raw_dataset.metadata.channels)
            if raw_dataset.metadata.channels
            else (raw_dataset.trials[0].shape[1] if raw_dataset.trials else 0)
        )
        # Plots A/B/C operate on raw_dataset/processed_dataset, which carry the
        # ORIGINAL labels (including noGesture at index 0). `class_names` here is
        # the REMAPPED list (post no-gesture drop), so resolving against it would
        # mislabel those datasets. Use the original class names for selection and
        # plots A/B/C; the prediction overlays below keep the remapped names.
        raw_class_names = raw_dataset.metadata.class_names
        selection = resolve_selection(vis.selection, raw_class_names, n_channels)

        # A: raw signal per selected gesture/channel
        for p in plot_raw_signal_by_class(
            raw_dataset,
            run_dir,
            class_names=raw_class_names,
            selection=selection,
            consistent_scale=vis.consistent_scale,
        ):
            self.tracker.log_artifact(p)

        # B: stage progression (one figure per (stage, gesture))
        if vis.stage_progression:
            stages = compute_stages(
                raw_dataset, self.cfg.signal_processing, viz_baseline
            )
            for p in plot_stage_progression(
                stages, selection, raw_class_names, vis.consistent_scale, run_dir
            ):
                self.tracker.log_artifact(p)

        # C: feature-over-windows (one figure per gesture)
        if vis.feature_over_windows.enabled:
            for p in plot_feature_over_windows(
                processed_dataset,
                self.cfg.windowing,
                selection,
                vis.feature_over_windows.features,
                vis.feature_over_windows.max_windows,
                raw_class_names,
                run_dir,
            ):
                self.tracker.log_artifact(p)

        # Prediction overlay + data flow: single-split only.
        if not is_cv:
            for part, ds, pred in (
                ("val", split.val, predictions.val),
                ("test", split.test, predictions.test),
            ):
                p = os.path.join(run_dir, f"predictions_on_signal_{part}.png")
                plot_predictions_on_signal(
                    ds,
                    pred,
                    self.cfg.windowing,
                    class_names,
                    path=p,
                    task=self.cfg.task.name,
                )
                self.tracker.log_artifact(p)

        if self.cfg.signal_processing:
            p = os.path.join(run_dir, "signal_processing_comparison.png")
            plot_signal_processing_comparison(raw_dataset, processed_dataset, path=p)
            self.tracker.log_artifact(p)

        if not is_cv:
            p = os.path.join(run_dir, "signal_data_flow.png")
            plot_signal_data_flow(
                raw_dataset,
                processed_dataset,
                split.test,
                predictions.test,
                self.cfg.windowing,
                class_names,
                path=p,
                task=self.cfg.task.name,
            )
            self.tracker.log_artifact(p)

        feature_specs = self.cfg.feature_extraction.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        if any("frequency_domain" in str(f.get("_target_", "")) for f in feature_specs):
            p = os.path.join(run_dir, "fft_spectrum.png")
            plot_fft_spectrum(
                processed_dataset,
                self.cfg.windowing,
                sampling_rate=processed_dataset.metadata.sampling_rate,
                path=p,
            )
            self.tracker.log_artifact(p)

    def _persist_artifacts(self, dataset, result) -> None:
        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        config_dir = os.path.join(run_dir, "config")
        os.makedirs(config_dir, exist_ok=True)
        resolved_path = os.path.join(config_dir, "resolved.yaml")
        save_resolved_config(self.cfg, resolved_path)
        self.tracker.log_artifact(resolved_path)

        report = render_report(result, self.cfg, dataset)
        if "json" in self.cfg.reporting.formats:
            JsonReporter().write(report, os.path.join(run_dir, "report.json"))
        if "html" in self.cfg.reporting.formats:
            HtmlReporter().write(report, os.path.join(run_dir, "report.html"))
