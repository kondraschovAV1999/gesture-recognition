from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace
from datetime import datetime

from hydra.utils import instantiate

from hand_gesture_recognition.configuration.loader import save_resolved_config
from hand_gesture_recognition.configuration.schema import ExperimentConfig
from hand_gesture_recognition.classical.adapters.sklearn import SklearnPredictionAdapter
from hand_gesture_recognition.data.fixtures import make_synthetic_dataset
from hand_gesture_recognition.data.splits.base import (
    assert_no_group_leakage,
    split_dataset,
)
from hand_gesture_recognition.evaluation.base import EvaluationResult, evaluate
from hand_gesture_recognition.logging_setup.config import get_logger
from hand_gesture_recognition.paths import resolve_path
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
        mlflow.sklearn.autolog()
        try:
            with stage(self.logger, "load_dataset"):
                dataset = self._load_dataset()
            with stage(self.logger, "analytics"):
                self._analytics(dataset)
            with stage(self.logger, "signal_processing"):
                dataset = self._signal_processing(dataset)
            with stage(self.logger, "split"):
                split = split_dataset(dataset, cfg.split)
                # Group leakage is only meaningful for group-based splits;
                # random/stratified splits intentionally mix subjects across
                # partitions (the smoke test uses strategy=random).
                if cfg.split.strategy in ("subject_wise", "session_wise", "predefined"):
                    assert_no_group_leakage(split)
            with stage(self.logger, "train"):
                if cfg.backend == "classical":
                    prediction = self._train_and_predict_classical(
                        split.train, split.test
                    )
                else:
                    prediction = self._train_and_predict_neural(split)
            with stage(self.logger, "evaluate"):
                result = evaluate(
                    prediction,
                    cfg.evaluation,
                    dataset_id=dataset.metadata.name,
                    split_info={"strategy": cfg.split.strategy},
                    class_names=dataset.metadata.class_names,
                )
            self.logger.info(f"Evaluation metrics: {result.metrics}")
            with stage(self.logger, "track_and_report"):
                self.tracker.log_dataset(dataset)
                self.tracker.log_params({"seed": cfg.seed, "backend": cfg.backend})
                self.tracker.log_results(result)
                self.tracker.log_confusion_matrix(result, dataset.metadata.class_names)
                self._persist_artifacts(dataset, result)
            self.tracker.finish("FINISHED")
            return result
        except Exception:
            self.tracker.finish("FAILED")
            raise

    def _load_dataset(self):
        if self.cfg.dataset.name == "synthetic":
            return make_synthetic_dataset(seed=self.cfg.seed)
        loader = instantiate(self.cfg.dataset.loader)
        # Resolve the dataset path to a full path anchored at the project root
        # (the config default is a relative path, which only works when the
        # process runs from the repo root).
        return loader.load(resolve_path(self.cfg.dataset.path))

    def _signal_processing(self, dataset):
        # Deterministic signal processing (rectify, filters, envelope) applied
        # to the raw continuous signal BEFORE splitting — safe since it learns
        # nothing from the data.
        if not self.cfg.signal_processing:
            return dataset
        steps = [instantiate(step_cfg) for step_cfg in self.cfg.signal_processing]
        trials = dataset.trials
        for step in steps:
            trials = [step.transform(t) for t in trials]
        return replace(dataset, trials=trials)

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

    def _train_and_predict_classical(self, train, test):
        from hand_gesture_recognition.classical.pipelines.windowing import (
            WindowingTransformer,
        )

        # shared windowing via the spec-mandated sklearn wrapper (Section 6.2)
        windower = WindowingTransformer(self.cfg.windowing)
        train_wd = windower.fit_transform(train)
        test_wd = windower.transform(test)

        # Learned preprocessing (3D, fit on train only) then feature extraction
        # (3D -> 2D) — both via hydra.utils.instantiate (ADR #18).
        X_train, X_test = train_wd.X, test_wd.X
        for step_cfg in self.cfg.classical.preprocessing:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_test = step.transform(X_test)
        for step_cfg in self.cfg.classical.features:
            step = instantiate(step_cfg)
            X_train = step.fit_transform(X_train)
            X_test = step.transform(X_test)

        model = instantiate(self.cfg.classical.model)
        t0 = time.perf_counter()
        model.fit(X_train, train_wd.y)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)
        duration = time.perf_counter() - t0

        adapter = SklearnPredictionAdapter()
        return adapter.to_prediction(
            windowed=test_wd,
            y_pred=y_pred,
            y_score=y_score,
            run_id=self.cfg.run_id,
            duration_s=duration,
        )

    def _train_and_predict_neural(self, split):
        import logging

        import lightning.pytorch as pl

        from hand_gesture_recognition.data.windowing.base import window_dataset
        from hand_gesture_recognition.neural.adapters.lightning import (
            LightningPredictionAdapter,
        )

        # Keep Lightning's own console chatter quiet; the framework logs stages.
        logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)

        pl.seed_everything(self.cfg.seed, workers=True)

        n_channels = len(split.train.metadata.channels)
        n_classes = len(split.train.metadata.class_names)

        # The DataModule's split/windowing args are runtime objects (containing
        # numpy arrays), so bind the config args via _partial_ first, then pass
        # the runtime objects directly (avoids OmegaConf trying to convert them).
        dm_partial = instantiate(self.cfg.neural.datamodule, _partial_=True)
        datamodule = dm_partial(split=split, windowing=self.cfg.windowing)
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

        t0 = time.perf_counter()
        trainer.fit(model, datamodule=datamodule)
        raw = trainer.predict(model, datamodule=datamodule)
        duration = time.perf_counter() - t0

        # Window the test partition (same shared windowing) to recover the
        # per-window metadata + y_true for the Prediction.
        test_wd = window_dataset(split.test, self.cfg.windowing)

        adapter = LightningPredictionAdapter()
        return adapter.to_prediction(
            windowed=test_wd,
            raw_predictions=raw,
            run_id=self.cfg.run_id,
            duration_s=duration,
        )

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
