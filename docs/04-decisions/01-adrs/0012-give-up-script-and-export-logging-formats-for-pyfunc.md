# ADR 0012: Give up on script and export logging formats; use pyfunc

## Status

Accepted

## Context

Serving a trained crowd-counting model requires a consistent, reproducible inference interface. MLflow supports multiple model logging formats:

1. **Script format** (`mlflow.pytorch.log_model` with `code_paths`): Serializes the model as TorchScript
2. **Export format** (`torch.jit.trace` or `torch.onnx.export`): Exports to a graph representation
3. **PyFunc format** (`mlflow.pyfunc.PythonModel`): A generic Python wrapper around custom inference logic

Early attempts used script and export formats to achieve "native" MLflow serialization and deployment.

## Decision

Abandon script and export formats in favor of a custom PyFunc wrapper (`ProductionPyTorchWrapper`). The wrapper loads the model from standard PyTorch checkpoints, manages preprocessing and postprocessing, and persists the entire `/src` folder alongside the model artifact.

## Consequences

- **Avoided format brittleness**: Eliminated unsupported/deprecated TorchScript and ONNX operator errors.
- **Preserved architecture flexibility**: The model can be refactored without worrying about graph trace compatibility.
- **Dynamic inference logic**: Custom preprocessing (cropping, normalization) and postprocessing (density map decoding, count extraction) are decoupled from the model graph.
- **Easier maintenance**: Changes to model architecture or inference logic only require editing Python code, not debugging graph conditions.
- **Full codebase versioning**: The `/src` folder is packaged with the model, ensuring all preprocessing utilities and helper functions are available at inference time.

## Rationale

The evolution through logging formats reveals the fundamental trade-off between convenience and flexibility:

1. **Script format failure**: TorchScript compilation exposed unsupported operations and deprecated PyTorch APIs. While some errors were resolved, maintaining compatibility across PyTorch versions became an ongoing burden.

2. **Export format failure**: Graph tracing (via `torch.jit.trace` or ONNX export) created implicit dependencies on the exact input shapes and operator behaviors during tracing. Every change to the model architecture required re-solving complex mathematical and graph-level conditions to ensure the traced graph remained valid. This tight coupling between code changes and graph state proved unsustainable.

3. **PyFunc success**: By wrapping the model in Python, we regained control over the entire inference pipeline:
   - Preprocessing is explicit and modifiable without graph concerns
   - The model is loaded as standard PyTorch state, not compiled code
   - Custom postprocessing (e.g., density map to count conversion) is straightforward
   - The `/src` folder is versioned with the artifact, preserving all utilities

## Implementation

The `ProductionPyTorchWrapper` implements `mlflow.pyfunc.PythonModel`:

```python
class ProductionPyTorchWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load params and model from artifacts
        self.params = BaseParams.from_json(context.artifacts['params'])
        self.model = CrowdCounter(params=self.params)
        self.model.load_state_dict(torch.load(context.artifacts['weights']))
        self.model.eval()
    
    def predict(self, context, model_input):
        # Custom inference with preprocessing/postprocessing
        # Returns count and density map
```

The model is logged with:
- `params.json`: Full parameter configuration
- `weights.pt`: Model state dictionary
- `code/src/`: Complete source code for reproducibility

## Related Decisions

- **ADR 0005** (Params object): The `BaseParams` is serialized and bundled with the model, ensuring inference uses the exact training configuration.
- **ADR 0004** (Class inheritance): The `ProductionPyTorchWrapper` instantiates `CrowdCounter` using the same initialization logic as training, maintaining consistency.
