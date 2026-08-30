# Model Architecture

The model is designed for crowd-density estimation in dense scenes. Instead of predicting a single scalar count, it produces a low-resolution or full-resolution density map whose spatial values represent local crowd intensity.

The count estimate is obtained by integrating the model output. This preserves local information and is better suited to highly crowded scenes than direct count prediction alone.
