# Project Overview

This repository implements a crowd-counting pipeline for dense scenes. The project combines dataset handling, preprocessing, model training, evaluation, and experiment tracking into a single workflow for estimating crowd density and total count from images.

## Core idea

The project uses density-map regression rather than pure detection or a single count prediction. This is important because crowd scenes are dense, highly occluded, and vary in scale across the image.

## Typical workflow

1. Load image and target density map pairs.
2. Apply dataset-specific preprocessing and augmentation.
3. Train a model to predict a density field.
4. Evaluate the output against ground-truth maps and count metrics.
5. Log runs in MLflow and compare experiment results.
6. Reuse or serve the best model through the inference path.

## Project intent

The repo is not a generic ML template. It is a focused research and engineering project built around understanding crowd distribution in challenging visual conditions.
