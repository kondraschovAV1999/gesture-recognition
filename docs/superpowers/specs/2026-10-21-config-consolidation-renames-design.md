# Config Consolidation + Renames

**Date:** 2026-10-21
**Status:** Proposed

## Context

The Hydra config has four trivial groups that don't need preset files
(`experiment`, `task`, `split`) plus a `features` group whose name is vague and a
`subject_wise` split strategy whose name is misleading (it actually splits on
`split_on`, which can point at `subject_ids` *or* `trial_ids`).

This change consolidates the trivial groups into `configs/config.yaml`,
renames the split strategy for clarity, and renames `features` →
`feature_extraction`.

## Goals

1. **Inline `experiment`, `task`, and `split` into `configs/config.yaml`** —
   delete `configs/experiment/`, `configs/task/`, `configs/split/`, and move
   their fields directly into `config.yaml`. (`split_on` is what selects the
   grouping; no preset files are needed.)
2. **Rename the `subject_wise` strategy → `group_wise`** (it splits on whatever
   `split_on` points to, not necessarily subjects).
3. **Rename `features` → `feature_extraction`** (config group, `FeaturesConfig`
   dataclass, `ExperimentConfig.features` field, `cfg.features.*` references).
4. Update docs and all tests for the new paths.

## Non-goals

- No change to split semantics, feature-extraction behavior, or the number of
  strategies (only the `subject_wise` value name changes).
- No consolidation of `session_wise` into `group_wise` (left as-is).

## Design

### 1. Inline `experiment`, `task`, `split`

`configs/config.yaml` becomes:

```yaml
defaults:
  - dataset: epn612
  - windowing: default
  - signal_processing: emg
  - gesture_crop: default
  - baseline: none
  - classical: random_forest
  - feature_extraction: default
  - neural: lstm
  - evaluation: basic
  - analytics: basic
  - cv: none
  - visualization: default
  - _self_

# experiment (was configs/experiment/default.yaml @package _global_)
backend: neural
seed: 42
deterministic: true
tracking:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: hand_gesture_recognition
reporting:
  formats: [json, html]

# task (was configs/task/*.yaml)
task:
  name: recognition

# split (was configs/split/subject_wise.yaml)
split:
  strategy: group_wise
  split_on: subject_ids
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  seed: 42

hydra: ...
```

Delete: `configs/experiment/default.yaml`, `configs/task/recognition.yaml`,
`configs/task/classification.yaml`, `configs/split/subject_wise.yaml`.

### 2. Rename `subject_wise` → `group_wise`

- `schema.py::SplitConfig`: `strategy` default `"subject_wise"` → `"group_wise"`;
  `validate()` strategy list updated.
- `data/splits/base.py::split_dataset`: branch `("subject_wise", "session_wise")`
  → `("group_wise", "session_wise")`.
- `runner/run.py`: `assert_no_group_leakage` condition
  `("subject_wise", "session_wise", "predefined")` → `("group_wise", "session_wise", "predefined")`.
- `schema.py` baseline validation: same `subject_wise` → `group_wise`.

### 3. Rename `features` → `feature_extraction`

- `schema.py`: `FeaturesConfig` → `FeatureExtractionConfig`; `ExperimentConfig.features`
  → `feature_extraction`; `cfg.features.validate()` → `cfg.feature_extraction.validate()`.
- `loader.py::_coerce`: `features=FeaturesConfig(...)` →
  `feature_extraction=FeatureExtractionConfig(...)`; import renamed.
- `configs/features/` → `configs/feature_extraction/`; `config.yaml` default
  `- feature_extraction: default`.
- `runner/run.py`: three `self.cfg.features.resolve_features(` →
  `self.cfg.feature_extraction.resolve_features(`.

### 4. Tests + docs

- Update `test_config_loader.py` / `test_config_schema.py` / `test_smoke.py` /
  any test referencing `classical.feature_type`, `features.*`,
  `subject_wise`, or the removed groups.
- Update `docs/experiments/configuration.md` (defaults list, `split` strategy
  table, `features` → `feature_extraction` section, remove `experiment`/`task`
  group references) and `docs/experiments/running.md` (override examples).

## Edge cases

- CLI `task=classification` no longer works; use `task.name=classification`.
- CLI `split=...` no longer works; use `split.strategy=...` / `split.split_on=...`.
- `split_on` default is `subject_ids` (the `trial_wise` value from an in-flight
  edit is dropped — `split_on` expects a field name).

## Out of scope

- Merging `session_wise` into `group_wise`.
- Changing split logic or feature extractors.
