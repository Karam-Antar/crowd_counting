# Dataset Preparation

The project expects paired image and density-map folders with a consistent ordering relationship.

## Typical setup

- images are stored in an images folder,
- density maps are stored in a ground-truth-npy folder,
- the paired files should be aligned by ordering,
- train/test or train/valid splits must be organized consistently.

## Important caveat

The code relies on matching list order rather than file-name matching, so a folder mismatch is a critical issue that can silently break training.
