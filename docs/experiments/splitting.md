# Splitting strategies

The framework splits a `Dataset` into train / val / test **before** windowing,
at one of two granularities: **trial-level** or **subject-wise** (group-level).

## Trial-level splits (same subject may appear in train and test)

| Strategy | Behavior |
|---|---|
| `random` | Shuffles *trials* and assigns them to train/val/test by ratio. |
| `stratified` | Like `random`, but preserves class proportions (scikit-learn `train_test_split` with `stratify=labels`). |
| `trial_wise` | Explicit alias of `random`; named for discoverability. |

With a trial-level split, different trials of the *same subject* can land in
train and test. This maximizes use of data and can improve within-subject
accuracy, but it risks **subject leakage** (the model sees that subject's
signal statistics during training), so reported accuracy may be optimistic for
unseen users.

## Subject-wise splits (a subject appears in exactly one partition)

| Strategy | Behavior |
|---|---|
| `subject_wise` | Shuffles *subjects* and assigns whole subjects to one partition. |
| `session_wise` | Same grouping mechanism keyed by `split_on` (e.g. session id). |
| `predefined` | Uses the dataset's own cohort markers (e.g. EPN612 `train:`/`test:` prefixes); test cohort held out entirely. |

Subject-wise splits guarantee no subject appears in more than one partition,
which is the honest setting for cross-user generalization. The runner asserts
`assert_no_group_leakage` for these strategies.

## Choosing a strategy

- **Cross-user generalization / realistic deployment** → `subject_wise` (or
  `predefined` for EPN612).
- **Within-user diagnostics / maximum data efficiency** → `random` /
  `stratified` / `trial_wise`.
- **Cross-validation** → set `cv.method=kfold` with `cv.group_folds=true`
  (subject-wise folds) or `cv.group_folds=false` (trial-level folds), or
  `cv.method=loso` (leave-one-subject-out).
