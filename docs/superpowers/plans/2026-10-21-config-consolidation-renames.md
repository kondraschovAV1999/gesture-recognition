# Config Consolidation + Renames Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inline the `experiment`/`task`/`split` config groups into `configs/config.yaml`, rename the `subject_wise` strategy to `group_wise`, and rename the `features` group to `feature_extraction`.

**Architecture:** The three trivial groups lose their preset files and their fields move into `config.yaml` (top-level for `experiment`, nested `task:` and `split:` for the others). The split strategy value `subject_wise` → `group_wise` across the schema, split logic, runner, and tests. The `features` config group/field/dataclass becomes `feature_extraction`.

**Tech Stack:** Python 3.10, Hydra, NumPy, scikit-learn, PyTorch Lightning, pytest.

## Global Constraints

- No behavior change — only config paths and names change. All existing tests stay green (after updating their config paths).
- `split_on` default is `subject_ids` (a field name, not a strategy name).
- `strategy` values: `group_wise | session_wise | random | stratified | trial_wise | predefined`.
- `feature_extraction.feature_type` values: `time_domain | frequency_domain | combined | custom` (unchanged semantics).
- Breaking config paths accepted: `task=classification` → `task.name=classification`; `split=...` → `split.strategy=...`; `features.*` → `feature_extraction.*`.
- Run tests from the repo root; the active Conda environment is used automatically.

---

### Task 1: Rename `features` → `feature_extraction`

**Files:**
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/configuration/loader.py`
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Modify: `configs/config.yaml`
- Rename: `configs/features/` → `configs/feature_extraction/`
- Test: `tests/unit/test_config_loader.py`, `tests/unit/test_config_schema.py`, `tests/integration/test_smoke.py`

- [ ] **Step 1: Rename in the schema**

In `schema.py`:
- Rename the class `FeaturesConfig` → `FeatureExtractionConfig` (class name only; its `features` list field stays).
- In `ExperimentConfig`, change `features: FeaturesConfig = field(default_factory=FeaturesConfig)` → `feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)`.
- In `validate_experiment_config`, change `cfg.features.validate()` → `cfg.feature_extraction.validate()`.

- [ ] **Step 2: Rename in the loader**

In `configuration/loader.py`:
- Change the import `FeaturesConfig,` → `FeatureExtractionConfig,`.
- In `_coerce`, change `features=FeaturesConfig(**d.get("features", {}))` → `feature_extraction=FeatureExtractionConfig(**d.get("feature_extraction", {}))`.

- [ ] **Step 3: Rename the config group directory**

Rename `configs/features/default.yaml` → `configs/feature_extraction/default.yaml` (via `git mv configs/features configs/feature_extraction`).

In `configs/config.yaml`, change `- features: default` → `- feature_extraction: default`.

- [ ] **Step 4: Rename the runner call sites**

In `runner/run.py`, replace all three occurrences of `self.cfg.features.resolve_features(` with `self.cfg.feature_extraction.resolve_features(`.

- [ ] **Step 5: Update tests**

- In `tests/unit/test_config_loader.py`: rename the four `test_features_*` functions to `test_feature_extraction_*`; change `features.feature_type=` overrides to `feature_extraction.feature_type=`; change `cfg.features.resolve_features` to `cfg.feature_extraction.resolve_features`; change `cfg.features.feature_type` to `cfg.feature_extraction.feature_type`.
- In `tests/unit/test_config_schema.py`: rename `test_features_config_defaults` → `test_feature_extraction_config_defaults` and `test_features_validate_rejects_custom_without_features` → `test_feature_extraction_validate_rejects_custom_without_features`; change `FeaturesConfig` → `FeatureExtractionConfig` in those two tests.
- In `tests/integration/test_smoke.py`: change `"features.feature_type=frequency_domain"` → `"feature_extraction.feature_type=frequency_domain"`.

- [ ] **Step 6: Commit**

```bash
git add src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/configuration/loader.py src/hand_gesture_recognition/runner/run.py configs/config.yaml configs/feature_extraction/default.yaml tests/unit/test_config_loader.py tests/unit/test_config_schema.py tests/integration/test_smoke.py
git rm -r configs/features
git commit -m task1-feature-extraction-rename
```

Note: the shell (Windows cmd.exe) mangles `git commit -m` messages with spaces — use the single-token message `task1-feature-extraction-rename`. If it errors, retry with `git commit -m "task1-feature-extraction-rename"`. Do NOT use `git add -A` or `git add .` (the working tree has unrelated uncommitted changes in `configs/signal_processing/emg.yaml`, `configs/split/subject_wise.yaml`, `configs/windowing/two_level.yaml`).

---

### Task 2: Inline `experiment`/`task`/`split` + rename `subject_wise` → `group_wise`

**Files:**
- Modify: `configs/config.yaml` (rewrite)
- Delete: `configs/experiment/default.yaml`, `configs/task/recognition.yaml`, `configs/task/classification.yaml`, `configs/split/subject_wise.yaml`
- Modify: `src/hand_gesture_recognition/configuration/schema.py`
- Modify: `src/hand_gesture_recognition/data/splits/base.py`
- Modify: `src/hand_gesture_recognition/runner/run.py`
- Test: `tests/unit/test_config_schema.py`, `tests/unit/test_cv.py`, `tests/unit/test_splits.py`, `tests/datasets/test_classical_full.py`

- [ ] **Step 1: Rewrite `configs/config.yaml`**

Replace the entire content of `configs/config.yaml` with:

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

# experiment (inlined from configs/experiment/default.yaml)
backend: neural
seed: 42
deterministic: true
tracking:
  tracking_uri: sqlite:///mlflow.db
  experiment_name: hand_gesture_recognition
reporting:
  formats: [json, html]

# task (inlined from configs/task/*.yaml)
task:
  name: recognition

# split (inlined from configs/split/subject_wise.yaml)
split:
  strategy: group_wise
  split_on: subject_ids
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2
  seed: 42

hydra:
  run:
    dir: output/test_runs/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: output/test_runs
  job:
    chdir: false
```

- [ ] **Step 2: Delete the removed config files**

```bash
git rm configs/experiment/default.yaml configs/task/recognition.yaml configs/task/classification.yaml configs/split/subject_wise.yaml
```

- [ ] **Step 3: Rename `subject_wise` → `group_wise` in the schema**

In `schema.py`:
- In `SplitConfig`, change `strategy: str = ("subject_wise"  # random | ...)` → `strategy: str = ("group_wise"  # random | ...)`.
- In `SplitConfig.validate()`, change `"subject_wise",` → `"group_wise",` in the strategies tuple.
- In the baseline validation, change `cfg.split.strategy in ("subject_wise", "session_wise", "predefined")` → `cfg.split.strategy in ("group_wise", "session_wise", "predefined")`.

- [ ] **Step 4: Rename in `data/splits/base.py`**

Change `if config.strategy in ("subject_wise", "session_wise"):` → `if config.strategy in ("group_wise", "session_wise"):`.

- [ ] **Step 5: Rename in `runner/run.py`**

Change the `assert_no_group_leakage` condition from `cfg.split.strategy in ("subject_wise", "session_wise", "predefined")` → `cfg.split.strategy in ("group_wise", "session_wise", "predefined")`.

- [ ] **Step 6: Update tests**

- `tests/unit/test_config_schema.py`: change `cfg.split.strategy = "subject_wise"` → `cfg.split.strategy = "group_wise"`.
- `tests/unit/test_cv.py`: change all three `SplitConfig(strategy="subject_wise", ...)` → `SplitConfig(strategy="group_wise", ...)`.
- `tests/unit/test_splits.py`: rename `test_subject_wise_split_has_no_leakage` → `test_group_wise_split_has_no_leakage` and `test_subject_wise_split_never_empty_partitions` → `test_group_wise_split_never_empty_partitions`; change both `strategy="subject_wise"` → `strategy="group_wise"`.
- `tests/datasets/test_classical_full.py`: change `strategy="subject_wise"` → `strategy="group_wise"` and `split_info={"strategy": "subject_wise"}` → `split_info={"strategy": "group_wise"}`.

- [ ] **Step 7: Commit**

```bash
git add configs/config.yaml src/hand_gesture_recognition/configuration/schema.py src/hand_gesture_recognition/data/splits/base.py src/hand_gesture_recognition/runner/run.py tests/unit/test_config_schema.py tests/unit/test_cv.py tests/unit/test_splits.py tests/datasets/test_classical_full.py
git commit -m task2-inline-configs-group-wise
```

Note: use the single-token commit message `task2-inline-configs-group-wise`. Do NOT use `git add -A` or `git add .`.

<!-- PART3 -->

---

### Task 3: Docs

**Files:**
- Modify: `docs/experiments/configuration.md`
- Modify: `docs/experiments/running.md`
- Modify: `docs/experiments/splitting.md`
- Modify: `docs/development/extending.md`
- Modify: `README.md`

- [ ] **Step 1: `docs/experiments/configuration.md`**

1. Replace the "current defaults list" code block with the new defaults + inlined fields (copy the `config.yaml` content from Task 2 Step 1 — the `defaults:` list plus the inlined `backend/seed/deterministic/tracking/reporting`, `task:`, and `split:` blocks, omitting the `hydra:` block or keeping it if already present).
2. In the `### split` section, change `subject_wise` → `group_wise` in the `strategy` table row and the paragraph below it (the parenthetical `(subject_wise, session_wise, predefined)` → `(group_wise, session_wise, predefined)`).
3. Rename the `### features` section heading → `### feature_extraction`, and change `features.features` → `feature_extraction.features` and `features.feature_type` → `feature_extraction.feature_type` in that section's prose.
4. Add a short note near the top (after the defaults list) that `experiment`, `task`, and `split` are inlined directly in `config.yaml` (no separate preset files), and that `task.name=` and `split.strategy=`/`split.split_on=` are used on the CLI.

- [ ] **Step 2: `docs/experiments/running.md`**

1. In the "Common overrides" table: change `split.strategy=subject_wise` → `split.strategy=group_wise`; add a row `| feature_extraction.feature_type=... | ... |` (or update any `features.` reference); change any `task=...`/`split=...`/`baseline=per_user` example that relied on a removed group preset.
2. In the Examples block: change `split.strategy=subject_wise` → `split.strategy=group_wise`; change `classical.feature_type=frequency_domain` → `feature_extraction.feature_type=frequency_domain` (if present); leave `split.strategy=predefined` and `dataset.name=synthetic` etc. unchanged.

- [ ] **Step 3: `docs/experiments/splitting.md`**

Change every `subject_wise` → `group_wise` (the strategy table row and the "Cross-user generalization → subject_wise" bullet). Keep `session_wise` and `predefined` as-is.

- [ ] **Step 4: `docs/development/extending.md`**

1. Update the defaults-list snippet to match the new `config.yaml` (remove `- split: subject_wise`, `- task: ...`, `- experiment: ...`, `- features: ...`; add `- feature_extraction: default`).
2. Change the `configs/split/subject_wise.yaml` reference to reflect that `split` is now inlined in `config.yaml` (and the strategy is `group_wise`).
3. Change the built-in-strategies list `random, stratified, subject_wise, session_wise, predefined` → `random, stratified, group_wise, session_wise, predefined`.

- [ ] **Step 5: `README.md`**

Search for `subject_wise`, `features=`, `feature_type`, `task=`, `split=`, or references to `configs/experiment`, `configs/task`, `configs/split` and update any that are stale (e.g. `split.strategy=subject_wise` → `group_wise`; the quick-start examples remain valid if they use `dataset.name=synthetic` and `split.strategy=predefined`).

- [ ] **Step 6: Commit**

```bash
git add docs/experiments/configuration.md docs/experiments/running.md docs/experiments/splitting.md docs/development/extending.md README.md
git commit -m task3-config-consolidation-docs
```

Note: use the single-token commit message `task3-config-consolidation-docs`.

---

## Self-Review

- **Spec coverage:** §1 inline experiment/task/split → Task 2 (Step 1 rewrite + Step 2 delete); §2 rename subject_wise → group_wise → Task 2 (Steps 3–6); §3 rename features → feature_extraction → Task 1; §4 tests+docs → Tasks 1–3.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code (config.yaml is fully specified; renames are explicit old→new).
- **Type consistency:** `FeatureExtractionConfig` is defined in Task 1 and referenced in Task 1 (loader, `_coerce`) and Tasks 1's tests with the same name; `feature_extraction` field is defined in Task 1 and read in Task 1 (runner `cfg.feature_extraction.resolve_features`) with the same name; `group_wise` is used consistently in the schema default, `validate()` list, `splits/base.py` branch, runner leakage check, baseline validation, and all updated tests/docs.

