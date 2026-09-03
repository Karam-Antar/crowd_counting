# ADR 0011: Try training on JHU-Crowd++ dataset

## Status

Deferred

## Context

The initial baseline model was trained on Shanghai Part A, a relatively small and controlled dataset. JHU-Crowd++ is a larger, more diverse dataset with 2500+ high-resolution images covering varied scenes, perspectives, and crowd densities.

Switching to JHU-Crowd++ offered the potential to:
- Train on a larger, more diverse dataset
- Improve generalization to real-world scenarios
- Establish a stronger baseline for future work

## Decision

Attempt training on the JHU-Crowd++ dataset, but defer intensive tuning on this dataset in favor of first exhausting improvements on Shanghai Part A.

## Rationale

While JHU-Crowd++ is attractive, the encoder-decoder architecture with modern backbones (EfficientNet, ConvNeXt) combined with 2500+ high-resolution images creates a substantial computational burden:
- Training time per epoch increases significantly
- GPU memory requirements grow with image resolution
- Hyperparameter tuning trials are slower and more expensive

Given fixed compute resources and the principle of iterating quickly on a smaller dataset first, the team decided to:
1. **Maximize learning on Shanghai Part A**: Extract the best possible performance on a smaller, faster-to-iterate dataset
2. **Defer JHU-Crowd++ tuning**: Use JHU-Crowd++ primarily for validation and cross-dataset robustness checks
3. **Transfer learning path**: Once Shanghai Part A reaches a plateau, transfer the best model to JHU-Crowd++ as a warm start

This strategy prioritizes efficient iteration and understanding the optimization landscape before committing to the computational overhead of a larger dataset.

## Consequences

- **Maintained iteration speed**: Experiments on Shanghai Part A remain fast, enabling rapid prototyping.
- **Deferred compute cost**: Avoided the overhead of tuning on a large dataset until necessary.
- **Cross-dataset validation**: JHU-Crowd++ remains available for evaluating generalization without being the primary tuning target.

## Future Work

Once ShanghaiTech Part A performance plateaus, the project should revisit JHU-Crowd++ with:
- Models pre-trained on Shanghai Part A
- Reduced learning rates for fine-tuning
- Potentially dataset-specific augmentation strategies

## Related Decisions

- **ADR 0010** (Unify directory structure): The unified structure enabled quick loading of JHU-Crowd++ for initial experiments.
- **ADR 0005** (Params object): The `dataset` parameter in `BaseParams` allows easy switching between datasets.
