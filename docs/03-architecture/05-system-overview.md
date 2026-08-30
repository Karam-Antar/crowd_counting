# System Overview

The project follows a typical research pipeline:

1. load a dataset,
2. prepare and transform the inputs,
3. train a density estimation model,
4. evaluate the predictions,
5. log the run,
6. serve or reuse the best model.

The codebase is intentionally separated by concern so that data handling, model logic, and experiment tracking remain understandable.
