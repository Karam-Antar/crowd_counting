# ADR 0004: Build class inheritance architecture for experiment tracking

## Status

Accepted

## Context

The project needs to support multiple experiment execution strategies:

1. **Standard single runs**: Train a model once with fixed hyperparameters, log the result to MLflow.
2. **Hyperparameter tuning**: Run multiple trials using Optuna, each spawning its own MLflow experiment context.

Both strategies share a large amount of common training logic:
- Building Lightning models from parameters
- Constructing data loaders and modules
- Setting up callbacks (early stopping, checkpointing, learning rate monitoring)
- Running the training loop via PyTorch Lightning
- Evaluating the model on validation and training sets
- Normalizing and logging metrics
- Packaging the final model payload

However, they differ in how they orchestrate runs and interact with the experiment tracking backend (MLflow). Additionally, the project aims to support multiple tracking backends and model registry strategies over time, requiring a pluggable design.

The naive approach—duplicating the training logic in each runner class—would create maintenance burden and make it difficult to evolve the training contract consistently.

## Decision

Build a class inheritance hierarchy using abstract base classes:

1. **`BaseExperimentRunner`**: Abstract base class defining the shared training workflow and common methods:
   - `_build_model()`: Constructs the Lightning model
   - `_evaluate_model()`: Runs validation and normalizes metrics
   - `_get_default_callbacks()`: Returns standard Lightning callbacks
   - `_run_training()`: Executes the full training loop and returns a `ModelPayload`
   - `run()`: Abstract method for subclasses to implement their execution strategy

2. **`StandardRunner`** (subclass): Implements single-run execution:
   - Accepts a pre-configured `BaseTracker` instance
   - Optionally loads from a checkpoint
   - Executes one training run with fixed parameters
   - Uploads the result to a model registry if configured

3. **`OptunaTuner`** (subclass): Implements hyperparameter tuning:
   - Accepts tracker and registry class types (not instances)
   - Spawns a new tracker instance for each Optuna trial
   - Runs the Optuna `objective()` function, which generates trial-specific parameters and executes training
   - Tracks the best payload across all trials for final registration

Additionally, use abstract base classes for plugin interfaces:

4. **`BaseTracker`**: Defines the contract for experiment tracking backends:
   - `get_logger()`: Returns a Lightning-compatible logger
   - `start_run()`: Initializes tracking context
   - `log_init()`: Records initial parameters
   - `log_results()`: Persists final metrics
   - `end_run()`: Closes the run context

5. **`BaseRegistry`**: Defines the contract for model artifact storage:
   - `upload_model()`: Persists a `ModelPayload` to the backend
   - `download_model()`: Retrieves a model from storage

Concrete implementations (e.g., `MLFlowTracker`, `MLFlowRegistry`) inherit from these bases and provide backend-specific logic.

## Consequences

### Positive

- **DRY (Don't Repeat Yourself)**: Training loop logic is defined once in `BaseExperimentRunner`, shared by both single-run and tuning strategies.
- **Separation of concerns**: Data handling, model logic, and experiment orchestration remain independent and testable.
- **Extensibility**: New execution strategies (e.g., distributed training, grid search) can be added by subclassing `BaseExperimentRunner`.
- **Pluggable backends**: Tracking and registry logic is decoupled from training logic via abstract interfaces. New backends (e.g., Weights & Biases, custom logging) can be added without modifying training code.
- **Clear contracts**: Abstract base classes document the expected interface for runners, trackers, and registries, making the codebase more understandable.
- **Reproducibility**: Centralized parameter handling, standardized metrics normalization, and consistent callback setup ensure runs are comparable and reproducible.

### Trade-offs

- **Abstraction overhead**: The inheritance hierarchy adds conceptual complexity. Developers must understand the base class contracts and subclass responsibilities.
- **Type hints and generics**: Using `type[torch.nn.Module]` and `type[BaseTracker]` requires more careful type management, though this improves static checking.
- **Callback customization**: Subclasses can override `_get_default_callbacks()`, but the base implementation assumes standard early stopping and checkpointing; highly specialized callbacks require further subclassing.

## Rationale

This design was chosen because:

1. **ML projects evolve**: The team knew that the training contract would be refined over multiple experimental runs. Centralizing this logic in a base class makes it easy to improve callbacks, metrics, or validation logic without touching both runner implementations.

2. **Multiple execution models**: Supporting both single runs and hyperparameter tuning within the same codebase requires shared logic but different orchestration. Inheritance naturally separates these concerns.

3. **Future-proofing**: The project aims to support multiple tracking backends and model registries. Abstract base classes ensure new implementations can be added without modifying existing runner code.

4. **Maintainability**: Future contributors can read `BaseExperimentRunner` to understand the training workflow, then focus on subclass-specific details (single-run vs. tuning orchestration).

## Related Decisions

- ADR 0001 (Density-map regression): The training logic must support density outputs; the model architecture and evaluation metrics reflect this choice.
- ADR 0002 (Geometry-adaptive targets): Preprocessing and label scaling are part of the centralized training workflow.
