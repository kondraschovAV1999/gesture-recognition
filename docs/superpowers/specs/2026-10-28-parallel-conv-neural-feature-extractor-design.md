# Parallel-conv neural feature extractor + FC head

Date: 2026-10-28
Status: approved (pending implementation)

## Context

The neural backend currently offers sequence/1D models (LSTM, DNN, CNN,
CNN-LSTM) operating on raw windows or handcrafted-feature sequences. We add a
2D parallel-convolution feature extractor (Inception-style, residual, LRN) that
consumes the **handcrafted small-window features** arranged as a 2D image, with
a fully-connected prediction head.

## Goals

1. Add `ParallelConvLayer` and `FeatureExtractionModule` (the 2D backbone from
   the user's reference code, cleaned and with configurable channel widths).
2. Add `ParallelConvModule(ClassifierModule)` — backbone + `AdaptiveAvgPool2d` +
   `Linear` head → logits (argmax handled by the shared base).
3. Input shape is configurable: `(batch, channels, small_windows, features)`,
   where `channels` = EMG channels, `small_windows` = per-big-window count,
   `features` = selected handcrafted features per small window. `Conv2d` +
   global pooling make the model independent of H/W.
4. Wire it into the two-level neural path with a dedicated grouped datamodule
   that reshapes `(n_small, F*C) → (C, n_small, F)`.

## Non-goals

- No change to the existing sequence models or the single-level path.
- The handcrafted feature extraction itself is unchanged; the CNN consumes its
  output.

## Design

### Model (`models/neural/parallel_conv.py`)

- `ParallelConvLayer(in_channels, bottleneck_channels=16, out_channels=18)` —
  4 branches (1x1→3x3, 1x1, maxpool→1x1, 1x1→5x5), concat → `4*out_channels`.
- `FeatureExtractionModule(in_channels, out_channels=18, lrn_size=5, alpha=1e-4,
  beta=0.75, k=1.0)` — 6 stacked `ParallelConvLayer` blocks with two residual
  connections and a final `LocalResponseNorm`. Output `(N, 4*out_channels, H, W)`.
- `ParallelConvModule(n_channels, n_classes, out_channels=18, optimizer=None)`:
  `FeatureExtractionModule(in_channels=n_channels)` → `AdaptiveAvgPool2d(1)` →
  `flatten` → `Linear(4*out_channels, n_classes)`. Subclasses `ClassifierModule`
  (inherits cross-entropy loss, argmax accuracy, macro-F1, optimizer,
  early-stopping).

### Datamodule (`models/neural/datamodule.py`)

- `ParallelConvGroupedEMGDataset(grouped, feature_specs, n_features, n_channels,
  max_small_per_big)` — computes per-big-window features
  `(max_small_per_big, F*C)` via the existing `FeatureExtractor.transform_grouped`,
  then reshapes each to `(C, max_small_per_big, F)` using
  `F = n_features // n_channels` (feature-major order → transpose).
- `ParallelConvGroupedEMGDataModule(windowed, feature_specs, n_features,
  n_channels, max_small_per_big, batch_size, num_workers)` — mirrors
  `GroupedEMGDataModule`, returning 3D `x` items.

### Strategy wiring (`models/neural/strategy.py`)

In the `GroupedWindowedPartitions` branch, detect the model by its `_target_`
(contains `"ParallelConvModule"`): use
`n_channels = windowed.train.big.X.shape[-1]` (EMG channels) and the
`ParallelConvGroupedEMGDataModule`; otherwise keep the current
`GroupedEMGDataModule` + `n_channels = n_features` path.

### Config (`configs/neural/parallel_conv.yaml`)

`datamodule` (unused for two-level) + `model._target_ = ...parallel_conv.ParallelConvModule`
(+ optional `out_channels`), `optimizer`, `early_stopping` (monitor `val_f1`),
`trainer`.

## Backward compatibility

- Additive only; existing models/configs unchanged.

## Testing

- Unit: `ParallelConvLayer`/`FeatureExtractionModule` forward shapes on a
  dummy `(N, C, H, W)`; `ParallelConvModule` forward → `(N, n_classes)`; the
  grouped datamodule reshapes to `(C, max_small_per_big, F)` with correct
  feature-major transpose (assert a known value lands in the right channel/feature).
- Integration: a two-level neural run with `neural=parallel_conv` produces a
  result with accuracy.

## Documentation

- One line in `docs/experiments/configuration.md` listing the new model.
