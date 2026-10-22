# Shared `features` Config Group

**Date:** 2026-10-21
**Status:** Proposed

## Context

Handcrafted feature extraction is currently configured through `ClassicalConfig`
(`classical.feature_type`, `classical.features`, `ClassicalConfig.resolve_features`).
But the **neural two-level** path also uses it — `_train_and_predict_neural_two_level`
calls `self.cfg.classical.resolve_features(...)` — so configuring neural features
requires reaching into the `classical.*` config group. This is confusing.

Move feature extraction to a first-class, backend-agnostic **`features`** config
group, consumed by both the classical pipeline and the neural two-level pipeline.

## Goals

- Add a top-level `features` config group + `FeaturesConfig`
  (`feature_type`, `features`, `resolve_features()`, `validate()`).
- `ClassicalConfig` keeps only `preprocessing` and `model`.
- Both `_train_and_predict_classical` and `_train_and_predict_neural_two_level`
  (and the `_visualize` FFT check) read `cfg.features.resolve_features(...)`.
- CLI: `features.feature_type=frequency_domain` (or a `features=...` preset).

## Non-goals

- No change to the feature extractors themselves (`TimeDomainFeatures`,
  `FrequencyDomainFeatures`).
- No change to single-level neural (which still feeds raw signal to the network).
- No backward-compat shim for `classical.feature_type` (breaking config path is
  accepted).

## Design

### 1. Schema — `FeaturesConfig`

`src/hand_gesture_recognition/configuration/schema.py`:

- Rename the module helpers `_CLASSICAL_FEATURE_TYPES` → `_FEATURE_TYPES`
  (keep `_TIME_DOMAIN_FEATURE` and `_frequency_domain_feature()` as-is).
- Add:

```python
@dataclass
class FeaturesConfig:
    feature_type: str = "time_domain"  # time_domain | frequency_domain | combined | custom
    features: List[Dict[str, Any]] = field(default_factory=list)

    def resolve_features(self, sampling_rate: float) -> List[Dict[str, Any]]:
        ...  # moved verbatim from ClassicalConfig.resolve_features

    def validate(self) -> None:
        ...  # moved verbatim; error messages now say "features.feature_type"
```

- `ClassicalConfig` becomes:

```python
@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(default_factory=lambda: {...})
```

  (drop `feature_type`, `features`, `resolve_features`, `validate`).
- `ExperimentConfig` gains `features: FeaturesConfig = field(default_factory=FeaturesConfig)`.
- `validate_experiment_config`: replace `cfg.classical.validate()` with
  `cfg.features.validate()`.

### 2. Loader

`configuration/loader.py::_coerce`: add `features=FeaturesConfig(**d.get("features", {}))`
and import `FeaturesConfig`.

### 3. Config files

- `configs/features/default.yaml`: `feature_type: time_domain`.
- `configs/config.yaml` defaults: add `- features: default`.
- `configs/classical/random_forest.yaml`: remove the `feature_type: time_domain` line.

### 4. Runner

`runner/run.py` — three `self.cfg.classical.resolve_features(...)` call sites
(`_train_and_predict_classical`, `_train_and_predict_neural_two_level`,
`_visualize`) become `self.cfg.features.resolve_features(...)`.

### 5. Tests

- `tests/unit/test_config_loader.py`: rename the three `test_classical_feature_type_*`
  tests to `test_features_*`, switch overrides `classical.feature_type=` →
  `features.feature_type=`, and `cfg.classical.resolve_features(...)` →
  `cfg.features.resolve_features(...)`. Add a `features=...` preset load test.
- `tests/integration/test_smoke.py`: `classical.feature_type=frequency_domain` →
  `features.feature_type=frequency_domain`.

### 6. Docs

`docs/experiments/configuration.md`: add a `### features` section; trim the
`feature_type` rows from the `### classical` section.

## Edge cases

- `features.feature_type=custom` requires a non-empty `features.features` list
  (validation, unchanged semantics).
- `resolve_features` error messages now reference `features.feature_type`.

## Testing

- Schema/loader tests cover `FeaturesConfig` defaults, `resolve_features`
  (time/frequency/combined/custom), and `validate` errors.
- Existing tests updated for the new config path; full suite stays green.

## Out of scope

- New feature extractors.
- Applying handcrafted features to single-level neural.
