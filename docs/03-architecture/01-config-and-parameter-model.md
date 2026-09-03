# Config and Parameter Model

Configuration has two layers. `src/config.py` owns project-wide paths and constants; `BaseParams` owns the model, data, loss, optimization, and experiment settings passed through the pipeline.

`BaseParams` includes image and crop sizes, augmentation, dataset and split settings, backbone and decoder choices, batch sizes, learning rate and scheduling, gradient controls, loss names and weights, focal parameters, Huber delta, monitoring, and artifact-related options.

The same object is consumed by the datamodule, model, Lightning module, experiment runners, loss functions, and serving wrapper. It supports dictionary and JSON serialization, flattened MLflow parameters, and Optuna trial suggestions. Serializing it with a model is required because label scaling, preprocessing, architecture, and loss settings form one compatibility contract.

Project-wide values include dataset roots, the reproducibility seed, label scaling, logging, and artifact paths. Keep paths environment-specific and keep run-specific behavior in `BaseParams` rather than introducing component-local magic numbers.
