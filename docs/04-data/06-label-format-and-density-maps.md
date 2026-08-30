# Label Format and Density Maps

The label format is a continuous density map rather than a scalar count or object bounding boxes.

The sum of a density map approximates the number of people in the scene. This preserves spatial structure and allows the model to learn how the crowd is distributed across the image.

This is a major design decision in the project and is the core reason the training pipeline is so sensitive to preprocessing and label scaling.
