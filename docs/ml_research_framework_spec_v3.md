# ML Research & Benchmark Framework — Implementation Specification (v3)

## Revision Notes (v3)

Two additions on top of v2, both new top-level sections. Everything from
Section 17 onward has shifted by two positions to make room for them —
navigate by content/heading, not by remembered numbers, if you're
comparing against v2.

- **New: Configuration Loading Module** (Section 18). v2 described what
  configuration must express (Section 17) but never named the runtime
  component responsible for composing, validating, resolving, and —
  critically — *replaying* it. This module is now the single place in
  the codebase allowed to call Hydra's composition API directly, and it
  formally defines the replay mode that "reproduce an experiment from its
  recorded configuration" (Purpose, Section 1) actually depends on.
- **New: Hyperparameter Tuning** (Section 22). Previously mentioned only
  as a one-line pointer to the Hydra Optuna Sweeper inside the sweeps
  subsection (v2 Section 17.2). It now has its own section covering
  search-space configuration, nested MLflow run tracking per trial,
  classical-vs-neural considerations, and how the tuning job reuses the
  Experiment Runner rather than introducing a separate execution path.
- Minor: Section 17.2 is trimmed to batch/comparison sweeps only, with a
  pointer to Section 22 for search-based tuning specifically.
- Minor: Experiment Tracking (Section 14), Research Comparison
  Requirements (Section 23), Testing Strategy (Section 27), Key
  Architectural Decisions (Section 26), and Definition of Done
  (Section 28) each got small additions tying in the two new sections.

## Revision Notes (v2)

- Windowing is a single, shared, backend-agnostic implementation
  (Section 6), instead of being implicitly duplicated inside the
  classical and neural pipelines.
- Splitting always happens before windowing, per partition (Section 5.3,
  Section 6.1), to prevent overlapping-window leakage across train/val/
  test boundaries.
- A `Prediction` object and backend adapter interface (Section 12) make
  the previously undefined "Prediction" box in the architecture diagram
  concrete, and give the `adapters/` directories in the project structure
  an explicit purpose.
- Reproducibility is anchored to a dataset content hash, not only a
  path/version label (Section 19).
- Structured/validated configuration is recommended over plain YAML,
  given the number of redundant, unit-convertible parameters (Section
  17).
- Cross-run comparison is a named component, not just "look in the
  MLflow UI" (Section 16, Section 23).
- Pluggable components are built via Hydra's `instantiate`/`_target_`
  pattern (Section 20, Section 21).
- Testing Strategy (Section 27) names the specific correctness
  properties — leakage, windowing math, cross-backend schema parity —
  that are easy to get silently wrong in this domain.

---

## 1. Purpose

Build a reusable research framework for controlled experiments on
time-series and tabular datasets, with a primary initial use case of EMG
and other physiological sensor data.

The framework must make it easy to:

- load different datasets without changing the experiment infrastructure;
- inspect and analyze a raw dataset before modeling;
- compose preprocessing and feature-extraction steps;
- run both classical ML and neural-network experiments;
- reuse the same evaluation and experiment-tracking infrastructure across
  both;
- vary sampling rate, windowing, preprocessing, features, model
  architecture, and training parameters through configuration;
- search over hyperparameters systematically, with the same tracking and
  reproducibility guarantees as a manually configured run;
- reproduce an experiment from its configuration and recorded artifacts;
- compare multiple experiments systematically.

The framework is an **experimental research/benchmarking platform**, not
a production inference platform.

---

## 2. Core Design Principles

### 2.1 Separate concerns

Keep the following responsibilities independent:

1. Dataset loading
2. Dataset splitting and windowing
3. Dataset analytics/exploration
4. Classical ML pipeline
5. Neural-network pipeline
6. Prediction normalization (backend adapters)
7. Evaluation
8. Experiment tracking
9. Logging and tracing
10. Reporting/comparison/visualization
11. Configuration composition and loading
12. Hyperparameter tuning orchestration

Do not force these responsibilities into one universal abstraction.

### 2.2 Use existing frameworks where their abstractions fit

Use:

- **scikit-learn** for classical ML pipelines, transformers, and
  estimators.
- **PyTorch Lightning** for neural-network models and training
  orchestration.
- **LightningDataModule** for neural-network dataset preparation,
  batching, and dataloaders — but not for windowing math itself (see
  Section 6).
- **Hydra** for hierarchical configuration and experiment composition,
  including `hydra.utils.instantiate` against `_target_` entries for
  every pluggable component (dataset loader, preprocessing step, feature
  extractor, model, split strategy), and its Optuna Sweeper plugin for
  hyperparameter tuning (Section 22).
- **MLflow** for experiment tracking, metrics, parameters, artifacts, and
  model/checkpoint tracking, including nested runs for tuning trials.
- A small amount of custom framework code to connect these components.

Do not create a custom general-purpose training framework, or a custom
hyperparameter-optimization engine, that duplicates sklearn, Lightning, or
Hydra Sweeper functionality.

### 2.3 Shared experiment concepts, different internal pipelines

Classical ML and neural-network experiments do not need identical
internals. They should share the concepts:

```
Dataset -> Split -> Windowing -> Training -> Prediction -> EvaluationResult -> Tracking/Reporting
```

but internally use different mechanisms for `Training`. **`Windowing` and
`Prediction` are not internal-only concepts — they are explicit,
implementation-shared stages** (Sections 6 and 12) that both backends
call through thin, backend-specific wrappers, rather than each
reimplementing. **Configuration composition and hyperparameter tuning are
likewise shared, implementation-level infrastructure** (Sections 18 and
22) that sit outside both backends rather than being duplicated per
backend.

---

## 3. High-Level Architecture

```text
                              HYDRA CONFIGURATION
                                      |
                                      v
                       Configuration Loading Module (Section 18)
                                      |
                                      v
                              Experiment Runner
                                      |
                                      v
                               Dataset Loader
                                      |
                                      v
                                   Dataset ------------------> Dataset Analytics (raw)
                                      |                                |
                                      v                                v
                        Train / Validation / Test Split       Stats / Plots / QC
                                      |
                                      v
                    Windowing (shared, applied per split partition)
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                    Classical ML               Neural Network
                    Pipeline (sklearn)          Pipeline (Lightning)
                    preprocessing ->            DataModule ->
                    features -> estimator       LightningModule -> Trainer
                         |                         |
                         v                         v
                  Classical Prediction        Neural Prediction
                       Adapter                     Adapter
                         |                         |
                         +------------+------------+
                                      |
                                      v
                         Prediction (shared schema, Section 12)
                                      |
                                      v
                                 Evaluation
                                      |
                                      v
                              EvaluationResult
                                      |
                         +------------+------------+
                         |                         |
                         v                         v
                      MLflow                   Reporting
                                                    |
                                                    v
                                         Cross-run Comparison
```

The Dataset Loader is an input provider, not a pipeline stage.

The Configuration Loading Module is the sole entry point for turning
Hydra config groups and CLI overrides — or a previously recorded
configuration — into the validated object the Experiment Runner consumes
(Section 18).

The Analytics layer is an inspection/diagnostic consumer of the raw
dataset and optionally of split/windowed/processed datasets.

Windowing sits **between splitting and the backend branch**, and is
implemented once (Section 6) — not once per backend.

Every prediction, regardless of backend, is normalized into a shared
`Prediction` object (Section 12) before it reaches Evaluation, which is
what allows Evaluation to remain fully backend-independent.

Reporting is a consumer of results and artifacts. It must not be embedded
into the modeling pipeline.

**A hyperparameter tuning job (Section 22) is a controlled batch of
Experiment Runner invocations over this same diagram, one per trial,
tracked as nested MLflow runs.** It is not a separate architecture, and
the Runner itself does not need to know whether it was launched directly
or by a sweeper.

---

## 4. Project Structure

```text
ml_framework/
├── src/                           # All framework/application source code
│   ├── data/
│   │   ├── loaders/
│   │   ├── datasets/
│   │   ├── splits/
│   │   └── windowing/              # Shared, backend-agnostic windowing (Section 6)
│   │
│   ├── preprocessing/
│   ├── features/
│   │
│   ├── prediction/                 # Shared Prediction dataclass + adapter interface (Section 12)
│   │   └── base.py
│   │
│   ├── classical/
│   │   ├── pipelines/
│   │   ├── models/
│   │   └── adapters/                # Implements prediction/base.py for sklearn outputs
│   │
│   ├── neural/
│   │   ├── datamodules/             # Calls src/data/windowing internally; no separate windowing logic
│   │   ├── modules/
│   │   ├── models/
│   │   ├── pipelines/
│   │   └── adapters/                # Implements prediction/base.py for Lightning outputs
│   │
│   ├── analytics/
│   ├── evaluation/                  # Consumes Prediction objects only — never backend-native output
│   ├── reporting/
│   │   └── comparison/               # Cross-run MLflow query + comparison tables (Section 16)
│   ├── tracking/
│   ├── logging/
│   ├── tracing/
│   ├── configuration/
│   │   ├── schema/                   # Structured/validated Hydra configs (Section 17)
│   │   └── loader.py                 # Configuration Loading Module: compose + replay + validate (Section 18)
│   ├── tuning/                       # Optuna pruning-callback integration, best-trial export (Section 22)
│   └── runner/
│
├── configs/                       # Hydra experiment/configuration definitions
│   ├── dataset/
│   ├── split/
│   ├── windowing/
│   ├── preprocessing/
│   ├── features/
│   ├── classical/
│   ├── neural/
│   ├── evaluation/
│   ├── analytics/
│   ├── hpo/                        # Search-space definitions for tuning jobs (Section 22)
│   └── experiment/
│
├── resources/                     # Static inputs and non-source project resources
│   ├── data/                       # Locally stored datasets or dataset subsets
│   ├── templates/
│   ├── metadata/
│   ├── schemas/
│   └── reference/
│
├── output/                        # Generated artifacts; normally not source-controlled
│   ├── visualization/
│   ├── models/
│   ├── logs/
│   ├── analytics/
│   ├── evaluation/
│   ├── comparison/
│   ├── tuning/                     # Search space, per-trial summaries, best trial config
│   └── test_runs/
│
├── docs/                          # Detailed codebase and research-framework documentation
│   ├── architecture/
│   ├── components/
│   ├── pipelines/
│   ├── datasets/
│   ├── experiments/
│   └── development/
│
├── tests/                         # Automated tests, not generated test results
│   ├── unit/
│   ├── integration/
│   ├── pipeline/
│   ├── datasets/
│   └── fixtures/
│
├── pyproject.toml
├── README.md
└── ...
```

### Directory responsibilities

- `src/`: all executable framework/application code.
- `src/data/windowing/`: the single windowing implementation used by both
  backends. Neither `classical/` nor `neural/` may contain independent
  windowing logic — only thin wrappers that call into this module.
- `src/prediction/`: the shared `Prediction` dataclass and abstract
  adapter interface. `classical/adapters/` and `neural/adapters/`
  implement it; nothing else should construct predictions directly.
- `src/configuration/`: the only module allowed to call Hydra's
  composition API directly. Every other component receives an already
  validated, resolved `ExperimentConfig` object — never a raw OmegaConf
  `DictConfig` (Section 18).
- `src/tuning/`: the small amount of glue code a tuning job needs beyond
  Hydra's Optuna Sweeper itself — pruning-callback integration for
  Lightning trials and best-trial export. The search space is
  configuration, not code, and lives in `configs/hpo/` (Section 22).
- `configs/`: Hydra configurations that describe how an experiment,
  analysis, or tuning job should run.
- `resources/`: datasets and other static resources consumed by the
  framework. Large external datasets should generally not be committed to
  Git; see Section 19 on content-hash-based dataset reproducibility.
- `output/`: generated artifacts from analytics, experiments, tests,
  models, logs, visualizations, cross-run comparisons, and tuning jobs.
- `docs/`: detailed architecture, component, pipeline, dataset,
  experiment, and developer documentation.
- `tests/`: automated test code and lightweight fixtures. See Section 27
  for the specific correctness properties this framework must test.
  Generated test reports or test-run artifacts belong in
  `output/test_runs/`, not in `tests/`.

### Output organization

Every experiment should have a unique `run_id`. Evaluation and other
generated artifacts should be grouped by run where practical. A
representative structure is:

```text
output/
├── analytics/
│   └── <dataset_id>/
│       ├── dataset_summary.json
│       ├── statistics/
│       ├── quality_control/
│       └── visualizations/
│
├── evaluation/
│   └── <run_id>/
│       ├── config/
│       │   └── resolved.yaml
│       ├── results/
│       ├── predictions/
│       └── artifacts/
│
├── models/
│   └── <run_id>/
│       └── ... checkpoints / serialized models ...
│
├── logs/
│   └── <run_id>/
│       ├── application.log
│       ├── training.log
│       └── errors.log
│
├── visualization/
│   └── <run_id>/
│       └── ... generated figures ...
│
├── comparison/
│   └── <comparison_id>/
│       ├── run_ids.json
│       ├── metrics_table.csv
│       └── config_diff.json
│
├── tuning/
│   └── <tuning_job_id>/
│       ├── search_space.yaml
│       ├── trials_summary.csv
│       └── best_trial/
│           └── resolved.yaml
│
└── test_runs/
    └── <run_id>/
        └── ... test artifacts ...
```

The exact filenames can evolve, but generated artifacts must be traceable
to a specific run, dataset analysis, comparison, or tuning job.

---

## 5. Dataset Layer

### 5.1 Dataset Loader

The loader is responsible for converting a dataset-specific source into
the framework's internal dataset representation.

Examples:

- NinaPro loader
- CSV/Parquet time-series loader
- MAT-file loader
- custom EMG dataset loader

The loader should handle dataset-specific file layouts and labels. It
should not contain model-specific training logic, and it should not
window the data — the loaded `Dataset` is continuous/trial-based, not
pre-windowed (see Section 6).

### 5.2 Internal Dataset Representation

Create a common dataset object containing, as appropriate:

- signals/features;
- labels/targets;
- subject identifiers;
- trial/session identifiers;
- channel names;
- sensor metadata;
- timestamps or sample indices;
- sampling frequency;
- units;
- class names;
- dataset name/version;
- content hash of the underlying source files (Section 19);
- provenance metadata.

For time-series datasets, preserve temporal structure. Do not flatten
sequences globally merely to satisfy a model.

The `Dataset` object at this stage represents the **raw or minimally
processed, unwindowed** signal per subject/session/trial. A separate,
explicitly named representation — `WindowedDataset` — is produced later,
per split partition, by the shared windowing stage (Section 6). Keeping
these as distinct types makes it visible in code whether a given
component expects continuous signal or fixed-size windows.

Example conceptual representation:

```text
Dataset
├── X                    # continuous / trial-based, not windowed
├── y
├── metadata
│   ├── sampling_rate
│   ├── channels
│   ├── subjects
│   ├── sessions
│   ├── units
│   └── content_hash
└── schema
```

### 5.3 Train/Validation/Test Splitting

Splitting must be explicit and configurable.

Support at minimum:

- random split;
- stratified split where appropriate;
- subject-wise split;
- session-wise split;
- predefined dataset splits.

For subject-dependent research, subject leakage must be avoided.

**Splitting always operates on the raw, unwindowed `Dataset` (Section
5.2), and always happens before windowing (Section 6).** This is a hard
ordering rule, not a suggestion: if overlapping/strided windows are
constructed before splitting, near-duplicate windows can end up on both
sides of a train/test boundary, leaking information and inflating
validation/test metrics. The pipeline is:

```
Dataset (raw) -> Split -> Windowing(train), Windowing(val), Windowing(test)
```

never the reverse.

The split strategy must be recorded in MLflow and in the experiment
metadata.

---

## 6. Windowing (Shared, Backend-Agnostic)

Windowing converts a continuous or trial-based signal into fixed-size (or
otherwise defined) segments with corresponding labels. Because window
size and stride are explicit comparison variables (Section 23:
"different window sizes"), and because both classical and neural
experiments must be comparable at a given window configuration, **windowing
must have exactly one implementation, used identically by both backends**
— not one implementation per backend.

### 6.1 Ordering rule

Windowing is applied **after** splitting, independently to each split
partition (Section 5.3). This prevents overlapping windows near a split
boundary from leaking between partitions.

### 6.2 Shared implementation

Implement windowing once, in `src/data/windowing/`, as a plain function or
class operating on the raw `Dataset` representation plus the windowing
configuration (Section 10). It must not import or depend on sklearn or
Lightning.

Both backends consume this same implementation through thin,
backend-idiomatic wrappers:

- **Classical**: wrapped as an sklearn-compatible transformer (e.g. a
  `WindowingTransformer` in `src/classical/pipelines/`) whose
  `fit_transform` calls into `src/data/windowing` internally, so it
  participates in a normal sklearn `Pipeline`.
- **Neural**: called directly inside `LightningDataModule.setup()` to
  build the windowed tensors before they are wrapped in a
  `torch.utils.data.Dataset`.

Do not reimplement windowing math separately for either backend. If a
backend needs different packaging of the windowed output (e.g. flattening
for classical vs. preserving `(time, channels)` for neural), that
packaging is a thin, separately named step **downstream** of the shared
windowing call — not a parallel windowing implementation.

### 6.3 Output representation

Windowing produces, for each split partition, a `WindowedDataset`:

```text
X_windowed: (n_windows, time_steps, channels)
y_windowed: (n_windows,)  or  (n_windows, time_steps)   # depending on label granularity
window_metadata: window start/end index, source trial/session/subject, timestamp
```

This is the representation both the classical and neural branches start
from.

---

## 7. Analytics Layer

The analytics layer is a first-class part of the framework, not an
afterthought.

Its purpose is to understand a dataset before and during experiments and
to provide reusable diagnostics.

### 7.1 Raw Dataset Analytics

Provide configurable analysis of the loaded raw (unwindowed) dataset.

At minimum support:

**Dataset-level statistics**

- number of samples/records;
- number of trials/windows, where applicable;
- number of subjects;
- number of sessions;
- number of classes;
- class distribution;
- number of channels;
- channel names;
- sampling frequency;
- signal duration;
- missing-value counts;
- duplicated records where meaningful;
- label distribution by subject/session;
- dataset size in memory/storage.

**Signal statistics**

For each channel, when applicable:

- mean;
- standard deviation;
- minimum/maximum;
- RMS;
- peak amplitude;
- quantiles;
- approximate noise statistics;
- saturation/clipping counts;
- missing/invalid samples.

**Time-series diagnostics**

Support plots such as:

- raw waveform examples;
- multi-channel waveform plots;
- signal distributions/histograms;
- channel-wise box plots;
- spectrograms;
- power spectral density;
- class distributions;
- duration distributions;
- subject-wise statistics;
- session-wise statistics;
- correlation matrices where appropriate.

For EMG specifically, include optional frequency-domain diagnostics and
channel activity summaries.

### 7.2 Processing Analytics

Analytics should optionally run after splitting, windowing, preprocessing,
and feature extraction. Examples:

- before vs. after filtering;
- before vs. after normalization;
- envelope visualization;
- window visualization (drawing from the shared windowing output, Section
  6.3);
- feature distributions;
- dimensionality summaries;
- feature correlations.

### 7.3 Quality-Control Checks

Analytics should detect and report common problems rather than silently
fixing them.

Examples:

- inconsistent sampling frequency;
- unexpected channel counts;
- missing channels;
- NaN/Inf values;
- extreme outliers;
- signal clipping;
- empty classes;
- severe class imbalance;
- malformed windows;
- label/data length mismatch;
- subject or session overlap between split partitions (see Section 27).

The framework should distinguish between:

- warning;
- error that prevents the experiment;
- informational diagnostic.

### 7.4 Analytics Outputs

Analytics results should be storable as experiment artifacts.

Store:

- machine-readable statistics (JSON preferred);
- tabular summaries where appropriate;
- generated figures;
- dataset schema summary;
- quality-control report.

Raw dataset analytics should be independently runnable from training.

---

## 8. Classical ML Pipeline

Use **scikit-learn Pipeline** as the core abstraction. It begins from the
already-windowed, per-partition `WindowedDataset` (Section 6.3) —
windowing itself is not implemented here.

```text
WindowedDataset (from Section 6, shared)
    |
    v
Preprocessing Transformer(s)
    |
    v
Feature Extraction Transformer(s)
    |
    v
ML Estimator
    |
    v
Classical Prediction Adapter (Section 12)
```

Typical examples:

- filtering -> feature extraction -> SVM;
- normalization -> handcrafted features -> Random Forest;
- statistical features -> logistic regression;
- preprocessing -> dimensionality reduction -> classifier.

Custom transformations should follow sklearn estimator/transformer
conventions wherever possible.

Important requirements:

- `fit` any learned preprocessing using training data only;
- apply fitted transformations to validation/test data;
- support parameter inspection;
- make components compatible with sklearn model selection where
  practical;
- maintain reproducibility via explicit random seeds;
- output predictions through the Classical Prediction Adapter (Section
  12), never as a raw `Pipeline.predict()` array passed directly to
  Evaluation.

Do not flatten time-series data prematurely. If a classical method
requires flattened data, make that conversion an explicit transformation
step downstream of windowing.

---

## 9. Neural Network Pipeline

Use **PyTorch Lightning** as the neural-network training framework.

Use **LightningDataModule** for dataset preparation and dataloaders — its
`setup()` calls the shared windowing implementation (Section 6) rather
than containing its own windowing logic.

Use **LightningModule** for model architecture, loss, optimizer
configuration, and training/validation/test steps.

```text
WindowedDataset (from Section 6, shared, built inside DataModule.setup())
       |
       v
LightningDataModule
       |
       v
Batching / tensor conversion / transforms
       |
       v
LightningModule
       |
       +-- DNN
       +-- CNN
       +-- LSTM
       +-- GRU
       +-- Transformer
       +-- other NN architectures
       |
       v
Neural Prediction Adapter (Section 12)
```

### 9.1 Shape handling

Neural-network models may require different input shapes. Examples:

- DNN: `(batch, features)`;
- LSTM/GRU: `(batch, time_steps, features)`;
- 1D CNN: commonly `(batch, channels, time)` depending on implementation.

The shared `WindowedDataset` representation (Section 6.3) preserves the
natural time-series structure. Model-specific input adaptation
(reshaping/transposing for a given architecture) belongs in the neural
model/data module layer, downstream of windowing.

Do not force DNN, LSTM, CNN, and Transformer models into identical tensor
shapes if that damages their natural representation.

### 9.2 LightningDataModule responsibilities

It should handle the neural-network-specific data lifecycle, including as
appropriate:

- invoking the shared windowing implementation (Section 6) per split;
- train/validation/test dataset objects;
- batch creation;
- DataLoader configuration;
- shuffle policy;
- worker configuration;
- optional augmentation;
- tensor conversion;
- device-aware data handling where appropriate.

Keep dataset discovery/loading (Section 5.1) separate from the DataModule
so that datasets remain framework-independent.

### 9.3 Lightning Trainer

Use Lightning Trainer for:

- training loop management;
- validation/testing;
- checkpointing;
- accelerator/device configuration;
- precision settings;
- callbacks, including Optuna's pruning callback when running as part of
  a tuning job (Section 22.4);
- logging integration;
- seeding via `pl.seed_everything(seed, workers=True)` (Section 19).

Avoid implementing a custom training loop unless a specific research
method requires it.

---

## 10. Sampling and Time-Series Configuration

Sampling-related configuration must be explicit because it affects the
meaning of every downstream experiment. The implementation and ordering
rules for windowing live in Section 6; this section defines the
configuration surface only.

### Dataset/signal properties

- sampling frequency / sampling rate (Hz);
- channel count;
- channel names;
- signal units;
- timestamp interpretation.

### Resampling

- whether resampling is enabled;
- target sampling frequency;
- resampling method;
- anti-aliasing behavior where applicable.

### Windowing

- window size in samples;
- window size in seconds;
- step size/stride in samples;
- step size in seconds;
- overlap percentage;
- padding strategy;
- boundary handling;
- whether labels are per-sample, per-window, or sequence-level.

The configuration must allow the user to specify either sample-based or
time-based values and derive the other representation where possible —
and this derivation must be validated, not merely documented (Section
17.1). For example:

```text
sampling_rate = 1000 Hz
window_size = 200 samples
=> window_duration = 0.2 seconds

step = 50 samples
=> step_duration = 0.05 seconds
=> overlap = 75%
```

If both a sample-based and a time-based value are supplied for the same
parameter and they are inconsistent given the sampling rate, config
resolution must fail with a clear error rather than silently picking one.

### Signal processing

Support configuration for, as applicable:

- high-pass filter;
- low-pass filter;
- band-pass filter;
- notch filter;
- filter type/order;
- cutoff frequencies;
- rectification;
- envelope extraction;
- normalization method;
- standardization method;
- per-channel vs global normalization.

For EMG, configurations should be able to express common processing
choices without hard-coding them into the framework.

---

## 11. Preprocessing and Feature Extraction

Preprocessing and feature extraction must be configurable and composable.

Examples of preprocessing components:

- band-pass filtering;
- notch filtering;
- rectification;
- envelope extraction;
- resampling;
- normalization;
- standardization;
- artifact removal.

(Windowing is intentionally not listed here — see Section 6.)

Examples of feature extraction:

**Time domain**

- RMS;
- mean absolute value;
- waveform length;
- variance;
- zero crossing;
- slope sign changes;
- statistical moments.

**Frequency domain**

- power spectral density features;
- median frequency;
- mean frequency;
- spectral entropy;
- band power.

**Time-frequency**

- wavelet features;
- spectrogram-derived features.

**Learned features**

Neural models may learn representations directly and therefore do not
necessarily require a handcrafted feature extraction stage.

The framework should support both:

```text
WindowedDataset -> Handcrafted Features -> Classical Model
```

and:

```text
WindowedDataset -> Raw Windowed Tensor -> Neural Model
```

---

## 12. Prediction Layer

### 12.1 Why this stage exists

`sklearn.Pipeline.predict()` / `predict_proba()` and Lightning's
`Trainer.predict()` return incompatible native formats: a numpy array
versus a list of batched tensors, potentially class indices versus
logits versus probabilities. Without a normalization stage, the
Evaluation layer (Section 13) would have to branch on backend internally,
which violates the "evaluation is framework-level and independent from
sklearn/Lightning" decision (Section 26).

### 12.2 `Prediction` object

Define a single framework-level dataclass in `src/prediction/base.py`:

```text
Prediction
├── run_id
├── sample_ids / window_ids
├── y_true                  # ground truth, when available
├── y_pred                  # class indices or regression values
├── y_score                 # optional: class probabilities / confidence
├── prediction_duration_s
└── backend                 # "classical" | "neural"
```

### 12.3 Backend adapters

Define an abstract interface in `src/prediction/base.py`
(e.g. `to_prediction(...) -> Prediction`), with two concrete
implementations:

- `src/classical/adapters/`: wraps `Pipeline.predict()` /
  `predict_proba()` output into a `Prediction`.
- `src/neural/adapters/`: wraps `Trainer.predict()` output —
  concatenating batches and applying whatever final activation
  (softmax/sigmoid) is needed to populate `y_score` consistently with how
  the classical adapter populates it.

Both adapters must produce `Prediction` objects with an **identical
schema** for a given task type (e.g. multi-class classification), so a
golden test can assert structural parity between backends (Section 27).

**Evaluation (Section 13) accepts only `Prediction` objects** — never
sklearn or Lightning outputs directly. Nothing outside
`classical/adapters/` and `neural/adapters/` should construct a
`Prediction`.

---

## 13. Evaluation Layer

Evaluation must be independent from both sklearn and Lightning, and
formally consumes only `Prediction` objects (Section 12).

Define a framework-level `EvaluationResult` concept containing:

- experiment identifier;
- dataset identifier;
- split information;
- the source `Prediction` (or a reference to it);
- ground truth where appropriate;
- metric values;
- confusion matrix where applicable;
- per-class metrics;
- aggregate metrics;
- execution metadata.

Support configurable metrics, including as appropriate:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- macro F1;
- weighted F1;
- ROC-AUC where applicable;
- PR-AUC where applicable;
- regression metrics when regression tasks are added.

Metrics must clearly define averaging strategy and class handling.

Do not bury evaluation logic inside individual model classes, and do not
special-case backend type inside the evaluator — the `Prediction` schema
should make that unnecessary.

---

## 14. Experiment Tracking

Use **MLflow** as the tracking backend.

Experiment tracking, logging, and tracing are related but distinct:

- **MLflow** records structured experiment-level information used to
  compare runs.
- **Logging** records operational information needed to understand what
  the application is doing.
- **Tracing** records execution flow and stage-level context so a failed,
  slow, or unexpected experiment can be diagnosed.

All three should be designed into the framework from the beginning.

Every experiment/run should record at minimum:

### Configuration

- full resolved configuration, as produced by the Configuration Loading
  Module (Section 18);
- dataset name/version;
- **dataset content hash** (Section 19) — not just name/path;
- sampling frequency;
- channel configuration;
- window size, stride/step, overlap (Section 10);
- split strategy (Section 5.3);
- preprocessing parameters;
- feature extraction parameters;
- model architecture/configuration;
- training configuration;
- evaluation configuration;
- framework/library versions where practical;
- random seeds (Section 19).

### Results

- all configured metrics;
- per-class metrics where applicable;
- validation/test results;
- training duration;
- inference/evaluation duration where useful.

### Artifacts

- model/checkpoint;
- resolved configuration;
- dataset analytics output;
- plots;
- confusion matrix;
- evaluation JSON;
- generated report;
- optional predictions (as serialized `Prediction` objects, Section 12).

### Hyperparameter tuning trials

When a hyperparameter tuning job (Section 22) is active, each trial is
tracked as a **child MLflow run nested under a parent tuning-job run**.
The parent run records the search-space definition and, once the job
completes, the best trial's identifier and resolved configuration. Child
runs remain individually inspectable and comparable through the same
mechanisms as any manually launched run (Section 16.1).

The goal is that a previous MLflow run — whether launched directly or as
one trial of a tuning job — contains enough information to understand
exactly what experiment was executed, and to identify the exact dataset
bytes it ran against.

---

## 15. Logging and Tracing

Logging and tracing are first-class framework capabilities. Every major
stage should emit enough diagnostic information to debug failures and
unexpected results without modifying experiment code.

### 15.1 Centralized logging

Provide centralized, configurable logging with at least:

- `DEBUG` for detailed diagnostics;
- `INFO` for normal experiment progress;
- `WARNING` for recoverable data/configuration issues;
- `ERROR` for failures;
- full exception stack traces for unexpected failures.

Logs should include contextual fields where applicable:

- `run_id`;
- dataset ID/version/content hash;
- experiment name;
- pipeline/backend type;
- stage/component name;
- model name;
- timestamp;
- process/thread information when useful.

Avoid logging raw datasets, complete signals, credentials, or other
unnecessary large/sensitive content.

Run-specific log files should be written to:

```text
output/logs/<run_id>/
├── application.log
├── training.log
└── errors.log
```

Console logging should remain available for interactive development.

### 15.2 Stage-level tracing

Trace at least these execution stages:

```text
configuration resolution (compose or replay)
data loading
data validation
raw dataset analytics
data splitting
windowing
preprocessing
feature extraction
model construction
training
prediction
prediction normalization (adapter)
evaluation
report generation
artifact persistence
run finalization
```

Each stage should expose:

- start/end status;
- execution duration;
- relevant run/configuration context;
- input/output summary where practical;
- exception information on failure.

A developer should be able to answer:

- Which stage failed?
- Which dataset/configuration was active?
- How long did each stage take?
- Where did the data shape change?
- Did the number of samples/windows/classes change unexpectedly?
- Which preprocessing or feature component produced the problematic
  output?
- Did model construction/training actually start?
- Did the classical and neural adapters produce `Prediction` objects with
  matching schema?
- Was this run a standalone experiment, or one trial of a tuning job —
  and if the latter, which parent run does it belong to?

### 15.3 Shape and schema tracing

For time-series experiments, optionally emit structured diagnostics such
as:

```text
raw:       (samples, channels)
split:     per-partition sample counts, subject/session IDs
windowed:  (windows, time_steps, channels)
features:  (windows, feature_count)
labels:    (windows,)
prediction: (n_samples,) y_pred, (n_samples, n_classes) y_score
```

For neural experiments, also record relevant tensor shapes at debug
level. This is particularly important for diagnosing DNN/LSTM/CNN input
mismatches.

Schema validation should occur at stage boundaries where practical. A
component should fail close to the point where it receives invalid data
instead of producing a later, opaque model error.

### 15.4 Error handling

When an experiment fails:

1. log a concise failure message;
2. preserve the full stack trace;
3. record the run ID and failing stage;
4. preserve the resolved configuration;
5. preserve artifacts already produced where safe;
6. mark the MLflow run as failed when an MLflow run exists — and, for a
   tuning trial, report the failure to the sweeper so the search can
   continue past it rather than aborting the whole job (Section 22.1).

Do not replace useful exceptions with generic errors that hide the
original failure.

### 15.5 Run correlation

Use the same `run_id` across:

```text
Resolved configuration (Configuration Loading Module, Section 18)
        ↓
logs / traces
        ↓
output artifacts
        ↓
Prediction / EvaluationResult
        ↓
MLflow run (standalone, or a child run within a tuning job)
```

This creates a complete traceability chain from intended configuration to
actual execution and final results.

### 15.6 Performance diagnostics

Tracing should make it possible to identify slow stages without requiring
a profiler for every run. At minimum collect elapsed time for:

- dataset loading;
- analytics;
- splitting;
- windowing;
- preprocessing;
- feature extraction;
- model initialization;
- training;
- inference;
- prediction normalization;
- evaluation;
- report generation.

More detailed profiling can be optional and enabled through
configuration.

---

## 16. Reporting Layer

Reporting is a consumer of `EvaluationResult` and associated artifacts. It
must remain independent from the pipeline implementation.

Potential reporters:

- JSON reporter;
- CSV reporter;
- HTML report;
- Markdown report;
- notebook-friendly output;
- MLflow artifact reporter;
- visualization reporter.

Multiple reporters may consume the same result.

```text
EvaluationResult
      |
      +----> JSON
      +----> HTML
      +----> Plots
      +----> MLflow
```

### 16.1 Cross-run comparison

Single-run reporting is not sufficient for requirement Section 23
("compare multiple experiments systematically"). Add a
`reporting/comparison` component whose job is to:

- accept a list of `run_id`s, or an MLflow experiment name/filter;
- query the MLflow tracking API for each run's resolved configuration and
  metrics;
- produce a tabular comparison (JSON/CSV/HTML) of metric values alongside
  a configuration diff between runs.

This pairs naturally with Hydra's multirun mode (`-m`, Section 17.2) for
batch comparisons, and with a hyperparameter tuning job's child runs
(Section 22.3) for comparing trials — both are just sets of `run_id`s to
this component. Output is written to `output/comparison/<comparison_id>/`
(Section 4).

Do not make reporting, including comparison, part of sklearn Pipeline or
Lightning training code.

---

## 17. Hydra Configuration

Hydra is the central configuration mechanism.

Configuration should be hierarchical and composable.

Suggested configuration groups:

```text
configs/
├── config.yaml
├── dataset/
├── split/
├── windowing/
├── preprocessing/
├── features/
├── classical/
├── neural/
├── evaluation/
├── analytics/
├── hpo/
└── experiment/
```

A complete experiment configuration should be able to specify:

- dataset;
- dataset split strategy;
- sampling frequency;
- channel configuration;
- resampling;
- window size;
- stride/step;
- overlap;
- preprocessing pipeline;
- feature pipeline;
- classical model OR neural model;
- batch size;
- epochs;
- learning rate;
- optimizer;
- scheduler;
- random seed;
- hardware/accelerator;
- evaluation metrics;
- analytics settings;
- MLflow settings;
- reporting outputs.

### 17.1 Structured configuration and validation

Given the number of parameters with redundant, unit-convertible
representations (window size in samples vs. seconds; stride vs. overlap
percentage), plain untyped YAML is not sufficient. Use Hydra's structured
configs (Python dataclasses registered via `ConfigStore`), or a
Pydantic-validated schema layer on top of resolved YAML, so that:

- type errors and missing required fields fail at config-resolution time;
- inconsistent redundant representations (Section 10) fail at
  config-resolution time, not mid-pipeline;
- every default has a documented, discoverable source (Section 19).

This is the schema the Configuration Loading Module (Section 18)
validates against — Section 17 defines what configuration must express;
Section 18 defines the component that enforces and resolves it at
runtime.

### 17.2 Sweeps and batches of experiments

Hydra should allow experiment variants to be selected from the command
line without changing source code, and its multirun mode (`-m`) is the
mechanism for generating batches of related, user-enumerated experiments
for comparison (Section 23).

For hyperparameter search specifically — where an outer loop varies
parameters according to an optimization strategy rather than an
explicit, user-enumerated list — see Section 22, Hyperparameter Tuning,
which also builds on Hydra multirun but adds a sweeper backend, a
search-space configuration group, and nested MLflow run tracking on top
of it.

Example conceptual experiment combinations:

```text
Dataset: NinaPro
Sampling: 1000 Hz
Window: 200 ms
Step: 50 ms
Preprocessing: band-pass + rectification + normalization
Features: RMS
Model: Random Forest
```

versus:

```text
Dataset: NinaPro
Sampling: 1000 Hz
Window: 200 ms
Step: 50 ms
Preprocessing: band-pass + normalization
Features: none
Model: LSTM
```

---

## 18. Configuration Loading Module

v1 and v2 described what configuration must express and how it should be
structured (Section 17), but never named the runtime component
responsible for turning that configuration into something the rest of
the framework can safely use. This section defines it explicitly:
`src/configuration/`.

### 18.1 Responsibilities

`src/configuration/` is the single source of truth for turning CLI/config-
group input — or a previously recorded configuration — into a validated,
fully resolved, typed configuration object. It is the **only** place in
the codebase allowed to call Hydra's composition API (`hydra.compose` /
the `@hydra.main` entry point) directly; every other component receives
an already-resolved `ExperimentConfig` object, never a raw OmegaConf
`DictConfig` and never Hydra APIs themselves.

Responsibilities:

- compose Hydra config groups + CLI overrides into a raw `DictConfig`;
- validate the composed config against the structured schema (Section
  17.1), failing with a clear, actionable error rather than propagating
  into the pipeline;
- resolve interpolations and derived values (e.g. window duration from
  window size + sampling rate, Section 10), so downstream components
  never re-derive them independently;
- convert the validated, resolved `DictConfig` into a typed
  `ExperimentConfig` dataclass — the only configuration object the rest
  of the framework consumes;
- attach a schema/framework version tag to every resolved config it
  emits (Section 18.4).

### 18.2 Two loading modes

The module must support:

- **Compose mode** (default): normal Hydra config-group + CLI-override
  composition, used when launching a new experiment or a new
  hyperparameter tuning trial (Section 22).
- **Replay mode**: load a previously persisted `resolved.yaml` (Section
  4, `evaluation/<run_id>/config/resolved.yaml`, or
  `tuning/<tuning_job_id>/best_trial/resolved.yaml`), re-validate it
  against the current schema, and reconstruct the same
  `ExperimentConfig` — bypassing default composition entirely. This is
  what makes "reproduce an experiment from its configuration" (Section 1)
  and the corresponding Definition of Done item (Section 28) actually
  implementable, rather than just implied.

Replay mode must not silently merge in newer defaults that didn't exist
at the time the original run was recorded; it reconstructs exactly what
was recorded, or fails clearly if the resolved config is no longer
schema-valid (Section 18.4).

### 18.3 Failure behavior

Configuration errors — missing fields, type mismatches, inconsistent
sample/time-based windowing values (Section 10), unknown `_target_`
references — must fail at load time, before dataset loading or training
starts, with an error message that names the offending field and its
location in the config tree. This is the enforcement point for the
validation guidance in Section 17.1; that section describes the intent,
this module implements it.

### 18.4 Schema versioning

Every resolved config persisted to `evaluation/<run_id>/config/
resolved.yaml`, or generated by a tuning trial (Section 22), is tagged
with a config schema version. When replaying an older config against a
newer schema, the loader should either migrate known-compatible changes
or fail explicitly naming the incompatibility — never silently
reinterpret an old field under new semantics.

---

## 19. Reproducibility

Every run must be reproducible as far as practical.

Record:

- random seed, applied consistently across Python's `random`, `numpy`,
  `torch` (via `pl.seed_everything(seed, workers=True)`), and every
  stochastic sklearn component's `random_state`;
- Python/package environment (e.g. lockfile hash or `pip freeze` captured
  as an MLflow artifact);
- git commit/version where available;
- complete resolved configuration, as produced and versioned by the
  Configuration Loading Module (Section 18);
- **dataset content hash** (e.g. SHA-256 over the raw source files used),
  in addition to dataset version/path and metadata — a path and a
  free-text version label do not guarantee the underlying bytes are
  unchanged months later. Where stricter data versioning is needed,
  consider DVC or an equivalent content-addressed store as an optional
  enhancement, not a hard requirement for Phase 1;
- train/validation/test split definition, including subject/session
  assignment (Section 5.3);
- model configuration;
- preprocessing configuration.

Avoid hidden defaults that materially affect research results. Any
default must be documented and recorded in the resolved configuration.

**Determinism tradeoff:** fully deterministic execution (e.g. disabling
cuDNN benchmark mode / nondeterministic GPU ops) can measurably reduce
training throughput. This must be an explicit, documented configuration
choice (`deterministic: true/false`), not a silent default in either
direction.

---

## 20. Experiment Runner

Create a small orchestration layer whose responsibility is to coordinate
the major stages.

Conceptually:

```text
1. Resolve configuration via the Configuration Loading Module — compose
   or replay mode (Section 18); this validates against the structured
   schema (Section 17.1)
2. Create `run_id` and initialize logging/tracing
3. Initialize MLflow tracking (as a standalone run, or a child run under
   a tuning job's parent run, Section 22.3)
4. Load dataset
5. Run dataset analytics/QC
6. Create train/validation/test split
7. Apply shared windowing per partition (Section 6)
8. Instantiate classical or neural pipeline via hydra.utils.instantiate
9. Train model
10. Generate predictions
11. Normalize predictions via the appropriate backend adapter (Section 12)
12. Run evaluation on the resulting Prediction
13. Record metrics/artifacts
14. Generate reports
15. Finish MLflow run
16. Persist/flush logs and tracing information for the run
```

The runner should not contain model-specific algorithms, and should not
hardcode branching per model or preprocessing choice. It resolves
configuration and instantiates and wires together components — every
pluggable piece (dataset loader, split strategy, preprocessing step,
feature extractor, model, adapter) should be constructed via
`hydra.utils.instantiate` against a `_target_` in configuration, so
adding a new one requires a new config entry, not a runner code change.

For a hyperparameter tuning job (Section 22), the Runner is invoked once
per trial by the Hydra Sweeper, exactly as if launched directly — the
Runner does not need to know, or special-case, whether it is running
standalone or as part of a search.

---

## 21. Classical vs Neural Selection

A single experiment configuration should explicitly state the experiment
backend, for example conceptually:

```text
backend: classical
```

or:

```text
backend: neural
```

This flag selects which pipeline factory the runner instantiates, but the
components underneath — model, preprocessing steps, features — should
still be built individually via `instantiate`/`_target_` (Section 20)
rather than a hardcoded if/else per component.

For classical experiments:

- construct sklearn `Pipeline` (windowing transformer + preprocessing +
  features + estimator, Section 8);
- call `fit`/`predict`;
- normalize output via the classical `Prediction` adapter (Section 12);
- evaluate using the framework evaluator (Section 13).

For neural experiments:

- construct `LightningDataModule` (internally calling shared windowing,
  Section 6);
- construct `LightningModule`;
- train using Lightning `Trainer`;
- obtain predictions via `Trainer.predict()`;
- normalize output via the neural `Prediction` adapter (Section 12);
- evaluate using the framework evaluator (Section 13).

The rest of the framework should remain unchanged.

---

## 22. Hyperparameter Tuning

Hyperparameter tuning is a first-class capability, not an ad hoc script
layered on top of the runner. It reuses the same Experiment Runner
(Section 20), Configuration Loading Module (Section 18), and MLflow
tracking (Section 14) as a single run — a tuning job is many Runner
invocations over a search space, not a separate execution path.

### 22.1 Mechanism

Use the **Hydra Optuna Sweeper plugin** as the default HPO backend, since
Hydra already owns configuration composition and multirun (Section 17.2)
for this framework — this avoids introducing a second, parallel
configuration/orchestration system for tuning alone. A search space is
expressed as part of the Hydra configuration (config group `configs/
hpo/`), as distributions over specific `_target_`-instantiated component
parameters (Section 20), not as a separate ad hoc dictionary maintained
outside Hydra.

Do not build a custom tuning loop. If a research need genuinely outgrows
the Hydra Sweeper (e.g. population-based training, or ASHA-style
aggressive early stopping across many parallel trials), treat that as a
deliberate, documented exception — not a default.

### 22.2 Search space definition

At minimum, support tunable parameters for:

- classical model hyperparameters (e.g. `n_estimators`, `C`, `max_depth`);
- preprocessing/feature parameters where it's meaningful to tune them
  (e.g. filter cutoff, window size — noting that varying window size
  changes the shared windowing output, Section 6, and must be
  re-windowed per trial, not cached across trials with different window
  configurations);
- neural training hyperparameters (learning rate, batch size, optimizer,
  scheduler, architecture width/depth).

Search space bounds/distributions live in configuration (`configs/hpo/`),
not in Python code, consistent with Section 17's general
configuration-ownership principle.

### 22.3 Per-trial tracking

Each HPO trial is a normal Experiment Runner invocation (Section 20) and
gets its own `run_id`, resolved configuration, logs, and `Prediction`/
`EvaluationResult`, exactly as a manually launched experiment would
(Section 15.5). Trials belonging to the same tuning job are recorded in
MLflow as **nested/child runs under a parent tuning-job run** (Section
14), so that:

- the parent run records the search-space definition and the tuning
  job's overall best trial/parameters;
- child runs remain individually inspectable, comparable, and
  reproducible through the same mechanisms as any other run (Section
  16.1, Section 20).

### 22.4 Classical vs. neural considerations

- **Classical**: a tuning trial fits/evaluates a single sklearn
  `Pipeline` instantiation per trial. Where a purely classical,
  non-comparative tuning pass is sufficient (e.g. quick baseline tuning
  without needing MLflow-level cross-backend comparison), sklearn's own
  `GridSearchCV`/`RandomizedSearchCV` may be used *inside* a single
  tracked run as an implementation detail — but this must not become the
  framework's general tuning mechanism, since it has no equivalent on
  the neural side and would break the "same tuning infrastructure across
  both backends" requirement (Section 1).
- **Neural**: a tuning trial trains one `LightningModule` via `Trainer`
  per trial. Support Optuna's pruning integration (reporting intermediate
  validation metrics to allow early termination of clearly unpromising
  trials, via a Lightning callback in `src/tuning/`) as an optional,
  configurable efficiency measure — not a default that silently discards
  trials.

### 22.5 Selecting and recording the best trial

At the end of a tuning job:

- identify the best trial by the configured primary metric (Section 13);
- persist the best trial's full resolved configuration (Section 18.2) as
  a first-class artifact of the parent tuning run and under `output/
  tuning/<tuning_job_id>/best_trial/` (Section 4), so it can be replayed
  directly (Section 18.2, replay mode) as a standalone experiment without
  re-running the search.

---

## 23. Research Comparison Requirements

The framework must make controlled comparisons possible.

Examples:

- same dataset, different preprocessing;
- same preprocessing, different features;
- same preprocessing/features, different models;
- classical ML vs. neural network;
- different window sizes;
- different sampling rates;
- subject-dependent vs. subject-independent split;
- different random seeds;
- same model/features, different hyperparameters, via a hyperparameter
  tuning job (Section 22).

An experiment should be identifiable by configuration rather than by
custom source-code modifications.

The framework should make it easy to run batches of related experiments
via Hydra multirun (Section 17.2) and compare their MLflow results using
the cross-run comparison component (Section 16.1). This includes
comparing trials within a single hyperparameter tuning job, since each
trial is tracked as an ordinary, individually addressable MLflow run.
Because windowing (Section 6) and prediction normalization (Section 12)
are shared implementations rather than per-backend reimplementations, a
"classical vs. neural at the same window size" comparison is guaranteed
to actually hold the window size constant.

---

## 24. Initial Implementation Scope

Implement the smallest useful vertical slice first.

### Phase 1 — Foundation

- Configuration Loading Module: compose mode and replay mode, with
  structured validation (Section 18);
- Hydra structured configuration (Section 17.1);
- one dataset loader;
- common `Dataset` representation;
- train/validation/test splitting, with a leakage check (Section 27);
- shared windowing implementation (Section 6);
- analytics with basic statistics and plots;
- sklearn classical pipeline;
- one classical model;
- `Prediction` object and classical adapter (Section 12);
- basic evaluation;
- MLflow tracking, including dataset content hash;
- JSON/HTML reporting.

### Phase 2 — Neural Pipeline

- `LightningDataModule` calling the shared windowing implementation;
- one `LightningModule`;
- one simple neural model, preferably DNN or LSTM;
- Lightning Trainer integration, with `seed_everything`;
- neural `Prediction` adapter (Section 12);
- same evaluator and MLflow tracking as Phase 1;
- cross-backend `Prediction`/`EvaluationResult` schema parity test
  (Section 27).

### Phase 3 — Research Extensions

- additional loaders;
- additional preprocessing methods;
- feature libraries;
- additional classical models;
- CNN/LSTM/Transformer models;
- richer analytics;
- automated batch experiments via Hydra multirun (Section 17.2);
- hyperparameter tuning via the Hydra Optuna Sweeper, with nested MLflow
  run tracking (Section 22);
- cross-run comparison reporting (Section 16.1);
- advanced reporting.

Do not implement the entire component library before validating the
architecture, including a successful configuration replay, with one
classical and one neural experiment sharing the same windowing and
prediction path.

---

## 25. Non-Goals

Do not initially build:

- a production model serving system;
- a custom deep-learning training engine;
- a custom experiment database;
- a universal tensor abstraction that hides all model-specific shapes;
- automatic preprocessing decisions without explicit configuration;
- a monolithic pipeline that forces sklearn and Lightning into the same
  execution model;
- a full data-versioning system (e.g. reimplementing DVC) — a content
  hash is the Phase 1 requirement; adopting DVC or similar is an optional
  later enhancement, not a build-it-yourself goal;
- a custom hyperparameter-optimization engine — use the Hydra Optuna
  Sweeper (Section 22) rather than implementing search or pruning logic
  directly.

---

## 26. Key Architectural Decisions

The implementation should preserve these decisions unless a strong
technical reason emerges to change them:

1. **sklearn Pipeline is the core abstraction for classical ML.**
2. **LightningDataModule + LightningModule + Trainer are the core
   abstraction for neural networks.**
3. **DataLoader is outside either modeling pipeline.**
4. **Analytics is a first-class layer and can run before and after
   processing.**
5. **Evaluation is framework-level and independent from
   sklearn/Lightning.**
6. **Reporting is a consumer, not a pipeline stage.**
7. **Hydra owns configuration and experiment composition, using
   structured configs and `instantiate`/`_target_` for every pluggable
   component.**
8. **MLflow owns structured experiment/run tracking and artifact
   tracking.**
9. **Centralized logging and stage-level tracing are required across the
   full execution path.**
10. **Every run must have a stable `run_id` that correlates
    configuration, logs, traces, artifacts, evaluation results, and
    MLflow.**
11. **Time-series metadata such as sampling rate, window size, stride,
    channels, and units must be explicit configuration/data metadata.**
12. **The framework should preserve temporal structure until a
    particular method explicitly requires reshaping or flattening.**
13. **Learned preprocessing must be fitted only on training data to
    prevent leakage.**
14. **The framework is designed for comparative research and
    reproducibility rather than production deployment.**
15. **Windowing has exactly one shared, backend-agnostic implementation;
    splitting always happens before windowing, per partition.**
16. **Classical and neural pipelines each normalize their output through
    a `Prediction` adapter with an identical schema before evaluation
    runs; Evaluation never touches backend-native output directly.**
17. **Dataset reproducibility is anchored to a content hash of the
    source files, not only a path or version label.**
18. **Pluggable components (models, preprocessing steps, features,
    split strategies) are constructed via Hydra's
    `instantiate`/`_target_` pattern rather than hardcoded dispatch
    logic.**
19. **Configuration loading, validation, resolution, and replay are
    centralized in a single Configuration Loading Module; no other
    component calls Hydra's composition API directly.**
20. **Hyperparameter tuning reuses the Experiment Runner and
    Configuration Loading Module rather than introducing a separate
    execution path; the Hydra Optuna Sweeper is the default HPO backend,
    with trials tracked as nested MLflow runs.**

---

## 27. Testing Strategy

Beyond ordinary unit/integration coverage, the following correctness
properties are high-value and easy to get silently wrong in this domain.
Test them explicitly:

- **Windowing math round-trip**: for a given sampling rate, window size
  in samples and in seconds (and stride/overlap) are mutually derivable
  and consistent (Section 10).
- **Split leakage**: for subject-wise and session-wise splits, assert
  zero subject/session ID overlap between train, validation, and test
  partitions — both immediately after splitting and after windowing
  (Section 5.3, Section 6.1).
- **Cross-backend `Prediction`/`EvaluationResult` schema parity**: run a
  small synthetic dataset through both the classical and neural
  pipelines at the same windowing/preprocessing configuration and assert
  the resulting `Prediction` and `EvaluationResult` objects share
  identical fields/shapes for the same task type (Section 12, Section
  13).
- **Config validation**: intentionally inconsistent sample-based and
  time-based windowing values are rejected at config-resolution time, not
  mid-run (Section 17.1).
- **Configuration replay parity**: loading a previously persisted
  resolved configuration via the Configuration Loading Module's replay
  mode (Section 18.2) reconstructs an `ExperimentConfig` identical to the
  one originally recorded, and — for a deterministic model and seed —
  produces matching predictions.
- **HPO trial isolation**: trials within a tuning job that vary a
  windowing parameter (e.g. window size) each produce a distinct
  `WindowedDataset` (Section 6) rather than silently reusing a cached
  windowing result from a different trial's configuration.
- **End-to-end smoke test**: one classical and one neural experiment, on
  a tiny fixture dataset, run through the full Runner (Section 20)
  without errors, producing an MLflow run, a resolved-config artifact,
  and a report.

Generated test artifacts belong in `output/test_runs/<run_id>/`,
consistent with Section 4.

---

## 28. Definition of Done for the First Working Version

The first implementation is considered successful when a researcher can:

1. Select a dataset entirely through configuration.
2. Run raw dataset analytics and obtain statistics plus plots.
3. Configure sampling frequency, window size, stride, and preprocessing
   without changing source code, with invalid combinations rejected at
   config-resolution time.
4. Run a classical ML experiment using an sklearn Pipeline built on the
   shared windowing output.
5. Run a neural-network experiment using
   LightningDataModule + LightningModule + Trainer, built on the same
   shared windowing implementation.
6. Confirm that classical and neural experiments at the same
   windowing/preprocessing configuration produce `Prediction` and
   `EvaluationResult` objects with identical schema.
7. Confirm a subject-wise or session-wise split configuration produces
   zero subject/session overlap between partitions.
8. View both experiments as MLflow runs, including dataset content hash,
   and compare their metrics/configuration — individually and via the
   cross-run comparison component.
9. Generate a machine-readable and human-readable report.
10. Inspect logs/traces for a run and identify failures, timing
    bottlenecks, or unexpected data-shape changes without modifying
    source code.
11. Load a previously recorded configuration via the Configuration
    Loading Module's replay mode (Section 18.2) and reproduce the
    experiment, including its dataset content hash, with no
    source-code modification.
12. Launch a small hyperparameter tuning job (Section 22) covering at
    least one classical and one neural model, and identify the best
    trial and its resolved configuration via MLflow.

The framework should demonstrate that adding a new dataset, preprocessing
method, feature extractor, or model primarily requires a new
component/configuration — instantiable via `_target_` — rather than
rewriting the experiment runner.
