# Diabetic-Retinopathy-Severity-Detection
Diabetic Retinopathy Severity Detection using EfficientNet-B0, Swin-Tiny, and ensemble model (EfficientNet-B0 + Swin-Tiny)

## Overview
This project implements a deep learning pipeline for diabetic retinopathy (DR) severity classification using retinal fundus images. The experiment compares three different approaches:

1. EfficientNet-B0
2. Swin Transformer Tiny (Swin-T)
3. Ensemble EfficientNet-B0 + Swin-T

The pipeline includes:
- Dataset downloading from Kaggle
- Dataset scanning and metadata generation
- Group-aware train/validation/test splitting
- Data preprocessing and augmentation
- Model training with transfer learning and fine-tuning
- Evaluation using multiple metrics
- Error analysis and qualitative visualization
- Model comparison and benchmarking

The dataset used is a combined DR dataset containing images from EyePACS, APTOS, and Messidor.

---

# Features

## Dataset Handling
- Automatic Kaggle dataset download
- Automatic extraction and dataset scanning
- Metadata creation using Pandas
- Group-aware splitting to avoid data leakage from augmented images

## Preprocessing
- Image resizing
- Normalization
- Online augmentation:
  - Horizontal flip
  - Vertical flip
  - Rotation
  - Color jitter
  - Random erasing

## Models
- EfficientNet-B0
- Swin Transformer Tiny
- Feature-level ensemble EfficientNet-B0 + Swin-T

## Training Strategy
- Transfer learning with pretrained ImageNet weights
- Two-stage training:
  - Head training
  - Fine-tuning
- Weighted loss for class imbalance handling
- Early stopping
- Learning rate scheduler
- Mixed precision training (AMP)

## Evaluation
Metrics used:
- Test Loss
- Accuracy
- Precision
- Recall
- F1-score
- Quadratic Weighted Kappa (QWK)
- Balanced Accuracy
- Latency
- FPS

## Visualization
- Training curves
- Loss curves
- QWK curves
- Confusion matrices
- Prediction confidence plots
- Correct vs incorrect prediction samples

---

# Project Structure

```text
project_root/
│
├── artifacts/
│   ├── best_model.pth
│   ├── best_model_final.pth
│   └── best_model.onnx
│
├── plots/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── qwk_curve.png
│   └── confidence_bar_*.png
│
├── predictions/
│   └── test_predictions.csv
│
├── logs/
│   └── tensorboard/
│
├── history.csv
├── evaluation_summary.json
├── classification_report.csv
└── confusion_matrix.csv
```

---

# Requirements

## Python Libraries
Install required dependencies:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pillow tqdm tensorboard
pip install kaggle
```

Optional:

```bash
pip install wandb
```

---

# Dataset Setup

## Step 1 — Download Kaggle API Key
Download `kaggle.json` from your Kaggle account:

Kaggle → Account → API → Create New API Token

## Step 2 — Upload kaggle.json
Upload `kaggle.json` to Google Colab.

## Step 3 — Dataset Download
The notebook automatically downloads:

```text
ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy
```

---

# How to Run

## Step 1 — Open Google Colab
Upload the notebook:

```text
dr_detection_final.ipynb
```

## Step 2 — Upload kaggle.json
Upload your Kaggle API key.

## Step 3 — Run Cells Sequentially
Run the notebook from top to bottom.

The pipeline will:
1. Download dataset
2. Extract dataset
3. Scan and build metadata
4. Split dataset
5. Create dataloaders
6. Train models
7. Evaluate models
8. Generate plots and reports
9. Compare models

---

# Training Configuration

## EfficientNet-B0
```python
img_size = 256
batch_size = 32
epochs_head = 5
epochs_ft = 10
lr_head = 1e-3
lr_ft = 2e-4
```

## Swin-T
```python
img_size = 256
batch_size = 32
epochs_head = 5
epochs_ft = 10
lr_head = 1e-3
lr_ft = 1e-4
```

## Ensemble
```python
img_size = 256
batch_size = 32
epochs_head = 5
epochs_ft = 10
lr_head = 1e-3
lr_ft = 1e-4
```

---

# Model Comparison

The experiment compares:

| Model | Strength |
|---|---|
| EfficientNet-B0 | Strong local feature extraction and efficient inference |
| Swin-T | Better global contextual understanding |
| Ensemble | Combines local and global features |

---

# Output Files

Important generated files:

| File | Description |
|---|---|
| history.csv | Training history |
| evaluation_summary.json | Final evaluation metrics |
| classification_report.csv | Precision, recall, F1-score per class |
| confusion_matrix.csv | Confusion matrix |
| test_predictions.csv | Prediction results per image |
| best_model_final.pth | Final trained model |
| best_model.onnx | ONNX exported model |

---

# Notes

- The dataset is imbalanced; weighted loss is used.
- Group-aware splitting prevents augmented image leakage.
- The ensemble model uses feature-level fusion.
- EfficientNet-B0 achieved the best overall balance between performance and efficiency.

---

# Future Improvements

Potential future work:
- Higher image resolution (384 or 512)
- More advanced preprocessing
- Medical-specific attention mechanisms
- Larger transformer architectures
- Better balancing strategies
- Self-supervised pretraining
- Explainability methods (Grad-CAM)

---

# Author

This project was developed for diabetic retinopath
