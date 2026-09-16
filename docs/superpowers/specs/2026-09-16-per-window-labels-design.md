# Per-Window Gesture Labeling + Full Classical Model Test — Design

**Date:** 2026-09-16
**Status:** Approved (pending user review)

## Problem

The current classical pipeline labels *every* window of a trial with that
trial's single `gestureName`. But each ~5 s EMG recording contains only a
short gesture execution; the samples before/after it are "rest". Stamping the
whole trial's label onto every window mislabels the rest portion, producing
noisy training data.

The EMG-EPN612 dataset provides ground-truth gesture timing per sample via
`groundTruthIndex` (a `[start, end]` tuple of EMG sample indices), so each
window can be labeled correctly.

## Requirements

1. **Per-window labels** — a window is labeled by where it falls relative to
   the gesture segment, not by the whole trial's label.
2. **Use `groundTruthIndex`** (`[start, end]`) for the gesture bounds.
3. **Overlapping windows** — already supported; window overlap stays
   configurable via `stride`/`overlap_percent` (default 75% overlap).
4. **"noGesture" is a class** — the model classifies 6 classes: `0` =
   noGesture/rest, `1..5` = the five gestures.
5. **Full classical model test** — a comprehensive end-to-end test of the
   classical pipeline.

## Design

### 1. Data model — gesture bounds on `Dataset`

Add two aligned per-trial arrays to `Dataset` (in
`hand_gesture_recognition/data/datasets/base.py`):

- `gesture_start: np.ndarray` — EMG sample index where the gesture begins;
  `-1` for rest trials.
- `gesture_end: np.ndarray` — EMG sample index where the gesture ends; `-1`
  for rest trials.

`_subset` in `splits/base.py` and `make_synthetic_dataset` in `fixtures.py`
must carry these fields through.

### 2. Loader extracts gesture bounds

`EPN612Loader.load` (in `data/loaders/epn612.py`) extracts, per training
sample:

- `groundTruthIndex` present → `(start, end)` from it.
- else `startPointforGestureExecution` present → `(start, len(emg))`.
- else (rest trial) → `(-1, -1)`.

*(`groundTruthIndex` lives at the sample level — `sample["groundTruthIndex"]` —
not inside `emg`, per data inspection.)*

### 3. Windowing computes per-window labels

`window_dataset` (in `data/windowing/base.py`) assigns, for window
`[s, s+W)` of trial `i`:

- rest trial (`gesture_start[i] == -1`) → `0` (noGesture)
- `s + W <= gesture_start[i]` → `0` (before gesture)
- `s >= gesture_end[i]` → `0` (after gesture)
- otherwise (window overlaps `[start, end]`) → the trial's gesture label (1–5)

`WindowedDataset.y` stays `(n_windows,)`; each entry is now the correct
per-window label (0 = rest, 1–5 = gesture).

### 4. Synthetic fixture

`make_synthetic_dataset` generates, per trial, a synthetic gesture segment
(e.g. a contiguous middle portion) and fills `gesture_start`/`gesture_end`
accordingly, so the per-window labeling path is exercised deterministically
in tests.

### 5. Full classical model test (real data, small subset — no synthetic)

`tests/datasets/test_classical_full.py` (auto-skips when
`resources/data/EPN612` is absent):

- Load a **small subject subset** (e.g. 6–8 subjects from `trainingJSON/`)
  using `EPN612Loader(subjects=[...])`.
- Subject-wise split → window (overlapping, per-window labels) →
  `TimeDomainFeatures` → `RandomForest` → predict → `SklearnPredictionAdapter`
  → `evaluate`.
- Assert:
  - windowed data contains both class `0` (rest) and gesture labels (1–5);
  - per-window labels match the ground-truth `[start, end]` bounds
    (rest windows → 0, gesture windows → gesture label);
  - `Prediction`/`EvaluationResult` schema is correct;
  - metrics are present and accuracy is above chance.

Run with:

```bash
python -m pytest tests/datasets/test_classical_full.py -v
```

## Testing strategy additions

- Windowing: a window fully before `start` → `0`; fully after `end` → `0`;
  overlapping/inside → gesture label (unit test with explicit bounds).
- Loader: `groundTruthIndex` used when present, `startPointforGestureExecution`
  fallback when absent (fixture-level test).
- Cross-backend parity (neural) and HPO remain deferred to Phase 2/3.

## Non-goals

- No neural pipeline changes (Phase 2).
- No change to evaluation/tracking/reporting (they already consume the
  per-window `Prediction` unchanged).
- No end-point estimation beyond what `groundTruthIndex` /
  `startPointforGestureExecution` provide.
