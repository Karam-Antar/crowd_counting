# Schema Contracts

The project depends on a clear understanding of the input and output interfaces used by the model and the training pipeline.

Schema documentation should include:

- image format and shape expectations,
- target density-map format,
- batch contract used by the dataloader,
- metadata expected by inference or serving components.

These contracts are important because they are easy to break during refactoring and are often the cause of training or evaluation instability.
