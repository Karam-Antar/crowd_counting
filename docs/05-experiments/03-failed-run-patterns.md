# Failed Run Patterns

This page captures recurring failure modes observed during experimentation.

Common patterns include:

- inconsistent image-target pairing,
- unstable label scaling,
- invalid or mismatched dataset splits,
- augmentation changes that hurt the underlying training contract,
- runs that are not comparable because both code and data changed at once.

Documenting these patterns helps future engineers avoid repeating the same mistakes or misreading the experiment history.
