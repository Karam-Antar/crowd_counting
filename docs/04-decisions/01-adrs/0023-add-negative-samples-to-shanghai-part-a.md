# ADR 0023: Add negative samples to ShanghaiTech Part A dataset

## Status

Accepted

## Context

The attention head with focal loss (ADR 0020) was designed to suppress false positives. However, it struggled to learn effectively on ShanghaiTech Part A alone because:
- ShanghaiTech Part A is a crowd-counting dataset: nearly all images contain crowds
- Negative samples (images with no crowds, or with only confusing patterns) were extremely rare
- The model had insufficient diverse negative examples to learn what "non-crowd" truly looks like

The attention head had the right loss function (focal loss) but insufficient supervision data. Without seeing enough variety in non-crowd regions, it could not generalize to suppress false positives in unseen images.

## Decision

Manually collect and inject **30–40 negative samples** from the internet into the ShanghaiTech Part A training set:

1. **Easy negatives**: Blank walls, empty rooms, landscape scenes, outdoor scenery with no people
2. **Hard negatives**: Images containing visual patterns that confuse the model:
   - Birds, flocks of birds (similar local density to crowds)
   - Vehicles, traffic scenes (repeating patterns)
   - Leaves, foliage, dense vegetation (complex texture)
   - Tiles, grids, repetitive structures (spatial aliasing risk)
3. **High resolution**: Prioritized images with high pixel density to match ShanghaiTech Part A's resolution characteristics

The negative samples were added to the training set and then split into train/validation subsets (maintaining the same distribution).

## Result: Breakthrough Improvement

Adding negative samples produced **a massive improvement**—the model's performance jumped noticeably, bringing it closer to the project's best baseline than ever before.

Key improvements:
- False positives in textured backgrounds dropped dramatically
- The attention head learned sharp, confident foreground masks
- Validation metrics (NAE, MAE) improved significantly
- Model generalization improved, especially on out-of-distribution scenes

## Why This Worked

1. **Explicit negative supervision**: Focal loss needed explicit examples of non-crowds. ShanghaiTech Part A alone provided almost none.

2. **Diversity of hard negatives**: Including confusing patterns (birds, vehicles, vegetation) forced the attention head to learn discriminative features that distinguish subtle crowd characteristics (human density, scale variation, spatial organization) from coincidental pattern matches.

3. **Easy negatives as anchors**: Blank and empty scenes provided clear negative examples that helped the model learn the extreme case of "no crowd."

4. **High resolution alignment**: Using high-resolution negative samples ensured they matched the visual characteristics of ShanghaiTech Part A, preventing domain shift.

## Dataset Composition After Addition

- **Original ShanghaiTech Part A**: ~3000 images (mostly crowds)
- **Added negatives**: 30–40 images (no crowds, or hard negatives)
- **New training distribution**: ~3–1.3% negatives (small but crucial)
- **Split into train/val**: Maintained ratio across splits

## Distinction from ADR 0003

This differs from **ADR 0003** (Remove mislabeled data), which focused on data quality (cleaning errors and mislabeled ground truth). ADR 0023 focuses on **data completeness**—the dataset was missing an entire class of samples needed for the attention head to learn.

## Implications for Loss Function Design

This decision validates a key insight from ADR 0020 (focal loss):
- Focal loss effectively handles imbalanced data **within a dataset**
- But it cannot create information that doesn't exist
- If an entire class is missing from training data, no loss function can compensate

Negative samples + focal loss together created a powerful combination:
- Focal loss weighted the learning toward hard negatives
- The negative samples provided the hard negatives for focal loss to emphasize

## Path Forward

This experience highlights the importance of **dataset-aware design**:
- Before designing complex loss functions, ensure the dataset contains adequate examples of all target behaviors
- Hard negative mining (ADR 0003 → clean, then ADR 0023 → augment with new negatives) is an iterative process
- The model's failure mode (false positives) should guide data collection strategy

For future work:
- Continue collecting edge-case negatives during model evaluation
- Explore systematic negative sampling strategies (e.g., model-driven hard negative mining)
- Consider active learning approaches to identify which negative samples would be most informative

## Related Decisions

- **ADR 0003** (Remove mislabeled data): Earlier data cleaning; this ADR extends it with strategic augmentation.
- **ADR 0015** (Add attention head): Architectural component designed to suppress false positives.
- **ADR 0020** (Replace BCE with focal loss): Loss function that emphasizes learning from hard negatives; negative samples provide the material for this emphasis.
- **ADR 0005** (Params object): Could parametrize negative sample mixing ratio (currently ~1.3%) for future tuning.
