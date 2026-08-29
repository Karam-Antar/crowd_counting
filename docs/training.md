# Crowd Counting Model Training & Evaluation Log

## Model Overview
- **Backbone / Architecture:** CSRNet (VGG-16 frontend + Dilated Conv backend)
- **Task Type:** Density Map Estimation & Integration Count
- **Target Deployment:** Real-time video stream / Static image analysis

## Datasets & Preprocessing
- **Training Set:** ShanghaiTech Part A (300 train / 182 test images)
- **Ground Truth Format:** Head point coordinates converted to density maps
- **Gaussian Kernel Settings:** Geometry-adaptive kernel ($k=3, \beta=0.3$)
- **Input Resolution:** Random crop $256 \times 256$, scale factor $1/8$ output

## Hyperparameters & Setup
- **Optimizer:** Adam (`lr=1e-5`, `weight_decay=1e-4`)
- **Loss Function:** MSE Loss on Density Maps
- **Batch Size:** 8
- **Epochs:** 150 (Early stopping triggered at epoch 112)
- **Seed:** 42

## Benchmark Results
| Dataset Split | MAE ↓ | MSE ↓ | Notes |
| :--- | :--- | :--- | :--- |
| ShanghaiTech A (Val) | 64.2 | 102.1 | Baseline model |
| ShanghaiTech A (Test)| 68.5 | 108.4 | Best checkpoint (epoch 112) |

## Artifact Locations
- **Best Weights File:** `/models/checkpoints/crowd_csrnet_best_mae68.pth`
- **ONNX Export:** `/models/exported/crowd_csrnet_fp16.onnx`
- **W&B Experiment Run:** `https://wandb.ai/my-org/crowd-counting/runs/run-id`