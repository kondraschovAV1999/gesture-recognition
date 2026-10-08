from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace
from datetime import datetime

import numpy as np
from hydra.utils import instantiate

from hand_gesture_recognition.configuration.loader import save_resolved_config
from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import (
    DataSplit,
    assert_no_group_leakage,
    split_dataset,
)
from hand_gesture_recognition.data.windowing.base import WindowedPartitions
from hand_gesture_recognition.evaluation.base import EvaluationResult, evaluate
from hand_gesture_recognition.logging_setup.config import get_logger
from hand_gesture_recognition.paths import resolve_path
from hand_gesture_recognition.prediction.base import PartitionPredictions
from hand_gesture_recognition.reporting.reporters import (
    HtmlReporter,
    JsonReporter,
    render_report,
)
from hand_gesture_recognition.tracking.mlflow_tracker import MlflowTracker
from hand_gesture_recognition.tracing.stage import stage
import mlflow


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        # Timestamped run_id (sortable, human-findable) + short unique suffix.
        self.cfg.run_id = cfg.run_id or (
            f"{cfg.backend}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        )
        # output root is read at construction time (tests set the env var after
        # import but before construction), so it must NOT be a module constant.
        self.output_root = os.environ.get("HAND_GESTURE_RECOGNITION_OUTPUT", "output")
        self.logger = get_logger(
            "hand_gesture_recognition.runner", run_id=self.cfg.run_id
        )
        self.tracker = MlflowTracker(cfg.tracking)

    def run(self) -> EvaluationResult:
        cfg = self.cfg
        self.tracker.start(cfg.run_id)
        # Auto-log the trained sklearn model/params into the active run
        # (classical backend); the tracker still owns the run lifecycle.
        if cfg.backend == "classical":
            mlflow.sklearn.autolog()
        else:
            mlflow.pytorch.autolog()

        try:
            with stage(self.logger, "load_dataset"):
                raw_dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(raw_dataset)
            with stage(self.logger, "gesture_crop"):
                cropped = self._gesture_crop(raw_dataset)

            # Global class set: no-gesture remap computed once from the FULL
            # dataset so every fold/partition shares the same mapping.
            class_names = cropped.metadata.class_names
            remap = None
            if not cfg.dataset.include_no_gesture:
                from hand_gesture_recognition.data.class_selection import (
                    build_no_gesture_remap,
                )

                remap, class_names = build_no_gesture_remap(cropped)
            dataset_name = cropped.metadata.name

            # Visualization-only baseline (fit on the FULL cropped dataset, a
            # diagnostic; training baselines are fit per partition in
            # _run_partition). Fitted only when visualization needs it.
            viz_baseline = None
            if cfg.baseline.enabled and cfg.visualization.enabled:
                from hand_gesture_recognition.preprocessing.baseline import (
                    BaselineRemoval,
                )

                viz_baseline = BaselineRemoval().fit(cropped)

            if cfg.cv.enabled:
                from hand_gesture_recognition.data.splits.folds import generate_folds

                with stage(self.logger, "split"):
                    folds = generate_folds(cropped, cfg.cv, cfg.split)
                fold_test_results = []
                for i, fold in enumerate(folds):
                    with stage(self.logger, f"fold_{i}"):
                        _, results = self._run_partition(
                            fold, class_names, remap, dataset_name
                        )
                    if results["test"] is not None:
                        fold_test_results.append(results["test"])
                from hand_gesture_recognition.evaluation.aggregate import (
                    aggregate_folds,
                )

                result = aggregate_folds(fold_test_results, cfg.run_id, dataset_name)
                self.logger.info(f"CV metrics (mean): {result.metrics}")
                split, predictions = None, None
            else:
                with stage(self.logger, "split"):
                    split = split_dataset(cropped, cfg.split)
                    if cfg.split.strategy in (
                        "subject_wise",
                        "session_wise",
                        "predefined",
                    ):
                        assert_no_group_leakage(split)
                predictions, results = self._run_partition(
                    split, class_names, remap, dataset_name
                )
                result = results["test"]

            # processed dataset for visualization (baseline first, then signal).
            processed_dataset = (
                viz_baseline.transform(cropped) if viz_baseline is not None else cropped
            )
            processed_dataset = self._apply_signal_processing(processed_dataset)

            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(cropped)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                if cfg.cv.enabled:
                    self._report_cv(result, fold_test_results, class_names)
                    run_dir = os.path.join(
                        self.output_root, "evaluation", self.cfg.run_id
                    )
                    self._write_cv_summary(fold_test_results, run_dir)
                else:
                    for part, r in results.items():
                        if r is None:
                            continue
                        self.tracker.log_results(
                            r, partition=(None if part == "test" else part)
                        )
                        self.tracker.log_confusion_matrix(
                            r, class_names, partition=part
                        )
                self._visualize(
                    cropped,
                    processed_dataset,
                    split,
                    predictions,
                    class_names,
                    viz_baseline,
                    is_cv=cfg.cv.enabled,
                )
                self._persist_artifacts(cropped, result)
            self.tracker.finish("FINISHED")
            return result
        except Exception:
            self.tracker.finish("FAILED")
            raise

    def _run_partition(self, split, class_names, remap, dataset_name):
        """Run baseline -> class selection -> signal -> window -> train -> evaluate
        on a single DataSplit. Returns (predictions, {part: result-or-None})."""
        baseline_remover = None
        if self.cfg.baseline.enabled:
            from hand_gesture_recognition.preprocessing.baseline import BaselineRemoval

            baseline_remover = BaselineRemoval().fit(split.train)
            split = DataSplit(
                train=baseline_remover.transform(split.train),
                val=baseline_remover.transform(split.val),
                test=baseline_remover.transform(split.test),
            )
        if not self.cfg.dataset.include_no_gesture:
            from hand_gesture_recognition.data.class_selection import drop_no_gesture

            split = DataSplit(
                train=drop_no_gesture(split.train, remap, class_names),
                val=drop_no_gesture(split.val, remap, class_names),
                test=drop_no_gesture(split.test, remap, class_names),
            )
        split = self._signal_processing(split)
        with stage(self.logger, "windowing"):
            windowed = self._window(split)
        with stage(self.logger, "train"):
            if self.cfg.backend == "classical":
                predictions = self._train_and_predict_classical(windowed)
            else:
                predictions = self._train_and_predict_neural(
                    windowed, n_classes=len(class_names)
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
        return predictions, results

    def _report_cv(self, result, fold_results, class_names):
        self.tracker.log_metrics(result.metrics)
        if self.cfg.cv.report_per_fold:
            for i, r in enumerate(fold_results):
                self.tracker.log_results(r, partition=f"fold{i}")
                self.tracker.log_confusion_matrix(
                    r, class_names, partition=f"fold{i}_test"
                )

    def _write_cv_summary(self, fold_results, run_dir):
        import json

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

    def _load_dataset(self):
        if self.cfg.dataset.name == "synthetic":
            return make_synthetic_dataset(seed=self.cfg.seed)
        loader = instantiate(self.cfg.dataset.loader)
        # Resolve the dataset path to a full path anchored at the project root
        # (the config default is a relative path, which only works when the
        # process runs from the repo root).
        return loader.load(resolve_path(self.cfg.dataset.path))

    def _apply_signal_steps(self, trials):
        if not self.cfg.signal_processing:
            return trials
        steps = [instantiate(step_cfg) for step_cfg in self.cfg.signal_processing]
        for step in steps:
            trials = [step.transform(t) for t in trials]
        return trials

    def _apply_signal_processing(self, dataset):
        return replace(dataset, trials=self._apply_signal_steps(dataset.trials))

    def _signal_processing(self, split):
        # Deterministic per-trial steps; applied per-partition after baseline /
        # class selection. Equivalent to the old pre-split application.
        return DataSplit(
            train=replace(
                split.train, trials=self._apply_signal_steps(split.train.trials)
            ),
            val=replace(split.val, trials=self._apply_signal_steps(split.val.trials)),
            test=replace(
                split.test, trials=self._apply_signal_steps(split.test.trials)
            ),
        )

    def _gesture_crop(self, dataset):
        if not self.cfg.gesture_crop.enabled:
            return dataset
        from hand_gesture_recognition.data.gesture_crop import crop_to_gesture

        return crop_to_gesture(dataset, self.cfg.gesture_crop)

    def _analytics(self, dataset) -> None:
        if not self.cfg.analytics.enabled:
            return
        from hand_gesture_recognition.analytics.stats import dataset_summary
        import json

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

    def _window(self, split) -> WindowedPartitions:
        from hand_gesture_recognition.data.windowing.base import (
            WindowedPartitions,
            window_dataset,
        )

        windowed = WindowedPartitions(
            train=window_dataset(split.train, self.cfg.windowing),
            val=window_dataset(split.val, self.cfg.windowing),
            test=window_dataset(split.test, self.cfg.windowing),
        )
        for part in ("train", "val", "test"):
            if getattr(windowed, part).X.shape[0] == 0:
                self.logger.warning(f"partition '{part}' has 0 windows after windowing")
        return windowed

    def _train_and_predict_classical(
        self, windowed: WindowedPartitions
    ) -> PartitionPredictions:
        X_train, X_val, X_test = windowed.train.X, windowed.val.X, windowed.test.X
        for step_cfg in self.cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_val = step.transform(X_val)
            X_test = step.transform(X_test)
        # Feature extractors are applied independently to the preprocessed
        # windowed signal and concatenated column-wise (FeatureUnion), so
        # multiple extractors (e.g. time + frequency) compose correctly.
        feature_specs = self.cfg.classical.resolve_features(
            self.cfg.dataset.sampling_rate
        )
        train_parts, val_parts, test_parts = [], [], []
        for step_cfg in feature_specs:
            step = instantiate(step_cfg)
            train_parts.append(step.fit_transform(X_train))
            val_parts.append(step.transform(X_val))
            test_parts.append(step.transform(X_test))
        X_train = np.concatenate(train_parts, axis=1)
        X_val = np.concatenate(val_parts, axis=1)
        X_test = np.concatenate(test_parts, axis=1)

        model = instantiate(self.cfg.classical.model)
        model.fit(X_train, windowed.train.y)

        adapter = SklearnPredictionAdapter()

        t0 = time.perf_counter()
        y_train_pred = model.predict(X_train)
        y_train_score = model.predict_proba(X_train)
        train_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_val_pred = model.predict(X_val)
        y_val_score = model.predict_proba(X_val)
        val_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_test_pred = model.predict(X_test)
        y_test_score = model.predict_proba(X_test)
        test_duration = time.perf_counter() - t0

        return PartitionPredictions(
            train=adapter.to_prediction(
                windowed=windowed.train,
                y_pred=y_train_pred,
                y_score=y_train_score,
                run_id=self.cfg.run_id,
                duration_s=train_duration,
            ),
            val=adapter.to_prediction(
                windowed=windowed.val,
                y_pred=y_val_pred,
                y_score=y_val_score,
                run_id=self.cfg.run_id,
                duration_s=val_duration,
            ),
            test=adapter.to_prediction(
                windowed=windowed.test,
                y_pred=y_test_pred,
                y_score=y_test_score,
                run_id=self.cfg.run_id,
                duration_s=test_duration,
            ),
        )

    def _train_and_predict_neural(
        self, windowed: WindowedPartitions, n_classes: int
    ) -> PartitionPredictions:
        import logging

        import lightning.pytorch as pl

        from hand_gesture_recognition.neural.adapters.lightning import (
            LightningPredictionAdapter,
        )

        # Keep Lightning's own console chatter quiet; the framework logs stages.
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

        pl.seed_everything(self.cfg.seed, workers=True)

        n_channels = windowed.train.X.shape[-1]

        # The DataModule's windowed args are runtime objects (containing numpy
        # arrays), so bind the config args via _partial_ first, then pass the
        # runtime objects directly (avoids OmegaConf trying to convert them).
        dm_partial = instantiate(self.cfg.neural.datamodule, _partial_=True)
        datamodule = dm_partial(windowed=windowed)
        # The optimizer config carries its own `_target_`; passing it through
        # instantiate would trigger recursive instantiation (Adam() without
        # params). Use _partial_ and pass it as a plain arg at call time.
        model_partial = instantiate(
            self.cfg.neural.model,
            n_channels=n_channels,
            n_classes=n_classes,
            _partial_=True,
        )
        model = model_partial(optimizer=self.cfg.neural.optimizer)
        # Early stopping callback (optional); passed as a runtime object, so
        # build the Trainer via _partial_ and pass callbacks at call time.
        callbacks = None
        if self.cfg.neural.early_stopping:
            callbacks = [instantiate(self.cfg.neural.early_stopping)]
        trainer_partial = instantiate(self.cfg.neural.trainer, _partial_=True)
        trainer = trainer_partial(callbacks=callbacks)

        trainer.fit(model, datamodule=datamodule)

        t0 = time.perf_counter()
        y_train_pred_raw = trainer.predict(
            model, dataloaders=datamodule.train_dataloader()
        )
        train_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_val_pred_raw = trainer.predict(model, dataloaders=datamodule.val_dataloader())
        val_duration = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_test_pred_raw = trainer.predict(
            model, dataloaders=datamodule.test_dataloader()
        )
        test_duration = time.perf_counter() - t0

        adapter = LightningPredictionAdapter()
        return PartitionPredictions(
            train=adapter.to_prediction(
                raw_predictions=y_train_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=train_duration,
            ),
            val=adapter.to_prediction(
                raw_predictions=y_val_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=val_duration,
            ),
            test=adapter.to_prediction(
                raw_predictions=y_test_pred_raw,
                run_id=self.cfg.run_id,
                duration_s=test_duration,
            ),
        )

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
        if not self.cfg.visualization.enabled:
            return
        # TEMPORARY: existing body, minus the reporting.visualize guard, will be
        # fully replaced in Task 12. Keep the current plot calls as-is for now.
        from hand_gesture_recognition.reporting.visualization import (
            plot_fft_spectrum,
            plot_predictions_on_signal,
            plot_raw_signal_by_class,
            plot_signal_data_flow,
            plot_signal_processing_comparison,
        )

        run_dir = os.path.join(self.output_root, "evaluation", self.cfg.run_id)
        os.makedirs(run_dir, exist_ok=True)
        for p in plot_raw_signal_by_class(raw_dataset, run_dir, None):
            self.tracker.log_artifact(p)
        if not is_cv:
            for part, ds, pred in (
                ("val", split.val, predictions.val),
                ("test", split.test, predictions.test),
            ):
                p = os.path.join(run_dir, f"predictions_on_signal_{part}.png")
                plot_predictions_on_signal(
                    ds, pred, self.cfg.windowing, class_names, path=p
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
            )
            self.tracker.log_artifact(p)
        feature_specs = self.cfg.classical.resolve_features(
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
