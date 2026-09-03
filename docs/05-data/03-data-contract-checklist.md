# Data Contract Checklist

Use this checklist before training, validation, or evaluation. It collects the assumptions already defined by the dataset and pipeline documentation.

## Dataset layout

- [ ] The configured dataset root points to the intended dataset.
- [ ] Each split contains an `images/` directory and a `ground-truth-npy/` directory.
- [ ] Shanghai Part A uses the documented `train_data/` and `test_data/` layout.
- [ ] JHU-Crowd++ follows the unified layout; a `valid/` directory may be used for validation.
- [ ] Image and density-map files are present in the same split.

## Pairing and labels

- [ ] The number of images equals the number of density maps.
- [ ] Image and label lists have the expected sorted parallel ordering.
- [ ] Density targets are stored as `.npy` arrays.
- [ ] Targets represent continuous density, not a single scalar count.
- [ ] The count represented by a target is obtained by summing its density values.
- [ ] Training label scaling is accounted for when interpreting counts and metrics.

## Dataset quality

- [ ] Invalid or mismatched image-density pairs have been removed.
- [ ] Any added negative samples are included before the normal train/validation split.
- [ ] Negative samples cover both easy empty scenes and hard non-human patterns where applicable.
- [ ] The negative-sample ratio is reviewed for possible underestimation bias.

## Split and preprocessing

- [ ] If a `valid` directory exists, its contents are used for validation.
- [ ] Otherwise, the split uses the documented deterministic 80/20 procedure with seed `42`.
- [ ] Training spatial transforms are applied jointly to images and density maps.
- [ ] Image normalization is not applied to density maps.
- [ ] Training density maps use the configured label scaler.
- [ ] Validation and test preprocessing remains deterministic unless five-crop mode is enabled.
- [ ] Padding preserves image-density alignment and satisfies the configured model stride.

## Batch and memory checks

- [ ] Validation and test batch sizes account for dynamic padding to the largest sample in a batch.
- [ ] Large outlier images have been considered before increasing evaluation batch size.
- [ ] Five-crop validation is enabled when full-resolution validation exceeds available memory.
- [ ] Fixed crop behavior is retained when comparing validation runs.

## When the contract changes

A change to layout, pairing, labels, scaling, splitting, or preprocessing can make experiments incomparable. Record such a change as a new decision and include the relevant parameter configuration with affected model artifacts.
