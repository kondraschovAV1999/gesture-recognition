# Shared `features` Config Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move handcrafted feature extraction out of `classical.*` into a top-level `features` config group, consumed by both the classical pipeline and the neural two-level pipeline.

**Architecture:** A new `FeaturesConfig` (`feature_type`, `features`, `resolve_features()`, `validate()`) replaces the feature-related fields on `ClassicalConfig` (which keeps only `preprocessing` and `model`). `ExperimentConfig` gains a `features` field, wired through `_coerce` and a `configs/features/` group. The three runner call sites read `cfg.features.resolve_features(...)`.

**Tech Stack:** Python 3.10, Hydra, NumPy, scikit-learn, PyTorch Lightning, pytest.

## Global Constraints

- Default `features.feature_type: time_domain` preserves current behavior; all existing tests stay green (after updating their config paths).
- `feature_type` values: `time_domain | frequency_domain | combined | custom`; `custom` requires a non-empty `features.features` list.
- `ClassicalConfig` keeps `preprocessing` and `model` only (no feature fields).
- Breaking change: `classical.feature_type=` overrides are replaced by `features.feature_type=` (accepted).
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: `FeaturesConfig` + loader + config files

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/configuration/loader.py`
- Modify: `configs/config.yaml`
- Create: `configs/features/default.yaml`
- Modify: `configs/classical/random_forest.yaml`
- Test: `tests/unit/test_config_schema.py` (append), `tests/unit/test_config_loader.py` (rewrite the 3 feature tests)

**Interfaces:**
- Produces: `FeaturesConfig(feature_type, features, resolve_features, validate)`.
- Produces: `ExperimentConfig.features: FeaturesConfig`; `ClassicalConfig` reduced to `preprocessing` + `model`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_config_schema.py`:

```python
def test_features_config_defaults():
    from hand_gesture_recognition.configuration.schema import FeaturesConfig
    f = FeaturesConfig()
    assert f.feature_type == "time_domain"
    assert f.features == []


def test_features_validate_rejects_custom_without_features():
    from hand_gesture_recognition.configuration.schema import FeaturesConfig
    f = FeaturesConfig(feature_type="custom")
    with pytest.raises(ConfigValidationError):
        f.validate()
```

In `tests/unit/test_config_loader.py`, REPLACE the three `test_classical_feature_type_*` functions with:

```python
def test_features_resolves_frequency():
    cfg = load_config(
        mode="compose",
        overrides=["features.feature_type=frequency_domain", "backend=classical"],
    )
    specs = cfg.features.resolve_features(cfg.dataset.sampling_rate)
    targets = [f["_target_"] for f in specs]
    assert any("FrequencyDomainFeatures" in t for t in targets)
    assert specs[0]["sampling_rate"] == cfg.dataset.sampling_rate


def test_features_resolves_combined():
    cfg = load_config(
        mode="compose",
        overrides=["features.feature_type=combined", "backend=classical"],
    )
    specs = cfg.features.resolve_features(cfg.dataset.sampling_rate)
    targets = [f["_target_"] for f in specs]
    assert any("TimeDomainFeatures" in t for t in targets)
    assert any("FrequencyDomainFeatures" in t for t in targets)


def test_features_rejects_unknown():
    with pytest.raises(ConfigValidationError):
        load_config(
            mode="compose",
            overrides=["features.feature_type=bogus", "backend=classical"],
        )


def test_features_defaults_to_time_domain():
    cfg = load_config(mode="compose")
    assert cfg.features.feature_type == "time_domain"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: FAIL (`FeaturesConfig` not defined; `features.feature_type=` is not a valid config path).

- [ ] **Step 3: Rework the schema**

In `src/hand_gesture_recognition/configuration/schema.py`, replace the block from `_CLASSICAL_FEATURE_TYPES = ...` through the end of `ClassicalConfig` with:

```python
_FEATURE_TYPES = ("time_domain", "frequency_domain", "combined", "custom")

_TIME_DOMAIN_FEATURE = {
    "_target_": "hand_gesture_recognition.features.time_domain.TimeDomainFeatures",
}


def _frequency_domain_feature(sampling_rate: float) -> Dict[str, Any]:
    return {
        "_target_": "hand_gesture_recognition.features.frequency_domain.FrequencyDomainFeatures",
        "sampling_rate": sampling_rate,
        "use_hann": True,
        "remove_dc": True,
    }


@dataclass
class FeaturesConfig:
    feature_type: str = (
        "time_domain"  # time_domain | frequency_domain | combined | custom
    )
    # Only used when feature_type == "custom" (an explicit list of _target_ entries).
    features: List[Dict[str, Any]] = field(default_factory=list)

    def resolve_features(self, sampling_rate: float) -> List[Dict[str, Any]]:
        """Return the concrete feature-extractor specs for this config.

        ``custom`` defers to the explicit ``features`` list; the other presets
        are built here so a single ``feature_type`` knob selects the
        representation. The sampling rate is injected from the dataset so the
        frequency extractor never needs it hardcoded in YAML.
        """
        if self.feature_type == "custom":
            return list(self.features)
        if self.feature_type == "time_domain":
            return [_TIME_DOMAIN_FEATURE]
        if self.feature_type == "frequency_domain":
            return [_frequency_domain_feature(sampling_rate)]
        if self.feature_type == "combined":
            return [_TIME_DOMAIN_FEATURE, _frequency_domain_feature(sampling_rate)]
        raise ConfigValidationError(
            f"features.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
        )

    def validate(self) -> None:
        if self.feature_type not in _FEATURE_TYPES:
            raise ConfigValidationError(
                f"features.feature_type={self.feature_type!r} not in {_FEATURE_TYPES}"
            )
        if self.feature_type == "custom" and not self.features:
            raise ConfigValidationError(
                "features.feature_type='custom' requires a non-empty features.features list"
            )


@dataclass
class ClassicalConfig:
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)
    model: Dict[str, Any] = field(
        default_factory=lambda: {
            "_target_": "hand_gesture_recognition.classical.models.random_forest.RandomForestEstimator",
            "n_estimators": 100,
            "random_state": 42,
        }
    )
```

- [ ] **Step 4: Add `features` to `ExperimentConfig`**

In `ExperimentConfig`, add the field after `classical`:

```python
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
```

- [ ] **Step 5: Update validation**

In `validate_experiment_config`, replace `cfg.classical.validate()` with `cfg.features.validate()`.

- [ ] **Step 6: Wire `_coerce`**

In `configuration/loader.py`, add `FeaturesConfig` to the import list (after `FeatureOverWindowsConfig`), and add `features=FeaturesConfig(**d.get("features", {}))` to the `ExperimentConfig(...)` constructor (after the `classical=` line).

- [ ] **Step 7: Update config files**

- Create `configs/features/default.yaml`:

```yaml
# @package features
feature_type: time_domain
```

- In `configs/config.yaml`, add `- features: default` after `- classical: random_forest`.

- In `configs/classical/random_forest.yaml`, delete the line `feature_type: time_domain   # time_domain | frequency_domain | combined | custom` (keep `preprocessing` and `model`).

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/unit/test_config_schema.py tests/unit/test_config_loader.py -v`
Expected: PASS (all tests, including the new/updated feature tests)

- [ ] **Step 9: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py configs/config.yaml configs/features/default.yaml configs/classical/random_forest.yaml tests/unit/test_config_schema.py tests/unit/test_config_loader.py
git commit -m task1-shared-features-config
```

---

### Task 2: Runner call sites + smoke-test update

**Files:**
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Modify: `tests/integration/test_smoke.py`

**Interfaces:**
- Consumes: `ExperimentConfig.features` (Task 1).

- [ ] **Step 1: Update the three runner call sites**

In `src/hand_gesture_recognition/runner/run.py`, replace all three occurrences of
`self.cfg.classical.resolve_features(` with `self.cfg.features.resolve_features(`.

They are in:
1. `_train_and_predict_classical`
2. `_train_and_predict_neural_two_level`
3. `_visualize` (the FFT-spectrum check)

- [ ] **Step 2: Update the smoke test**

In `tests/integration/test_smoke.py`, replace the override `"classical.feature_type=frequency_domain"` with `"features.feature_type=frequency_domain"`.

- [ ] **Step 3: Commit**

```bash
git add src/hand_gesture_recognition/runner/run.py tests/integration/test_smoke.py
git commit -m task2-shared-features-runner
```

---

### Task 3: Docs

**Files:**
- Modify: `docs/experiments/configuration.md`

- [ ] **Step 1: Add a `### features` section and trim `### classical`**

In `docs/experiments/configuration.md`, add a new `### features` section before the `### classical` section:

```markdown
### `features`

| Field | Values | Default |
|---|---|---|
| `feature_type` | `time_domain` \| `frequency_domain` \| `combined` \| `custom` | `time_domain` |
| `features` | explicit `_target_` list (when `feature_type: custom`) | `[]` |

Selects the handcrafted feature representation, used by the classical pipeline
and the neural two-level pipeline:

| `feature_type` | extractors |
|---|---|
| `time_domain` | `TimeDomainFeatures` (RMS, std, min, max, mean, max/mean, min/max) |
| `frequency_domain` | `FrequencyDomainFeatures` (total power, peak/mean/median frequency) |
| `combined` | both, concatenated column-wise |
| `custom` | whatever you list in `features.features` |
```

In the `### classical` section, remove the `feature_type` / `features` table rows and the following "`feature_type` picks the handcrafted feature representation" table (they moved to `### features`), leaving only `preprocessing` and `model`.

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/configuration.md
git commit -m task3-shared-features-docs
```

---

## Self-Review

- **Spec coverage:** §1 schema (`FeaturesConfig`, reduced `ClassicalConfig`, `ExperimentConfig.features`, validation) → Task 1; §2 loader `_coerce` → Task 1; §3 config files → Task 1; §4 runner call sites → Task 2; §5 tests → Tasks 1–2; §6 docs → Task 3.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code.
- **Type consistency:** `FeaturesConfig` is defined in Task 1 and referenced (`cfg.features`) in Task 2 with the same name; `resolve_features(sampling_rate)` signature is identical to the moved method; `features.feature_type` config path is used consistently across tasks; `_FEATURE_TYPES` is defined once and used only in `FeaturesConfig`.
