# 🗑️ Garbage Classification Model

[![CI](https://github.com/vintendohuynh/GarbageClassificationModel/actions/workflows/ci.yml/badge.svg)](https://github.com/vintendohuynh/GarbageClassificationModel/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Tests](https://img.shields.io/badge/Tests-16%20passed-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GarbageMLModel__CS549-blue)](https://huggingface.co/vincenthuynh/GarbageMLModel_CS549)

A Convolutional Neural Network (CNN) built with modern TensorFlow / Keras 3 to classify waste into 6 categories: **cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**.

Trained on the [Kaggle Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) (~2,500 images) with data augmentation, achieving ~92% validation accuracy.

### 🔗 [View on Hugging Face](https://huggingface.co/vincenthuynh/GarbageMLModel_CS549)

---

## 📁 Modular Python Architecture

```
├── src/
│   ├── __init__.py      # Package exports and version
│   ├── model.py        # Modern Keras CNN architecture, save/load (.keras & .h5)
│   ├── dataset.py      # Kaggle download, 80/20 train/val split, ImageDataGenerators
│   ├── train.py        # CLI trainer with configurable hyperparameters
│   ├── predict.py      # Inference pipeline & CLI for single images or directories
│   └── utils.py        # Evaluation curves, confusion matrix, and reporting
├── tests/              # Comprehensive pytest test suite (16 tests)
├── requirements.txt    # Project dependencies
└── GarbageMLModel_CS549_cleaned.ipynb  # Original Colab notebook
```

---

## 🚀 Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training via CLI

Download dataset from Kaggle and train automatically:

```bash
python -m src.train --download --epochs 50 --batch-size 32 --model-path GarbageMLModel_CS549.keras
```

Or train from a local dataset directory:

```bash
python -m src.train --data-dir ./data --epochs 30 --plot-history history.png
```

CLI options:
- `--download`: Automatically fetch dataset via `kagglehub`.
- `--data-dir`: Root dataset folder with class subdirectories (or explicit `--train-dir` and `--val-dir`).
- `--epochs`: Training epochs (default: `50`).
- `--batch-size`: Batch size (default: `32`).
- `--lr`: Adam learning rate (default: `0.001`).
- `--dropout`: Optional dropout rate before output layer (e.g. `0.2`).
- `--early-stopping`: Patience for early stopping callback (e.g. `5`).
- `--plot-history`: Path to save training curves plot.

---

### 3. Inference via CLI

Classify a single image:

```bash
python -m src.predict --image example.jpg --model GarbageMLModel_CS549.keras --top-k 3
```

Output:
```text
Predictions for: example.jpg
--------------------------------------------------
1. glass            92.40% (index 1)
2. plastic           5.10% (index 4)
3. metal             1.80% (index 2)
--------------------------------------------------
```

Classify an entire directory or output as JSON:

```bash
python -m src.predict --dir ./test_images --json
```

---

### 4. Python API Usage

```python
from src.predict import GarbagePredictor

# Initialize predictor
predictor = GarbagePredictor(
    model_path="GarbageMLModel_CS549.keras",
    classes_path="classes.json",  # optional, defaults to 6 Kaggle classes
)

# Predict single image
predictions = predictor.predict("example.jpg", top_k=3)
for p in predictions:
    print(f"{p['class']}: {p['confidence'] * 100:.2f}%")
```

---

## 🧪 Running Tests

Run the automated test suite with pytest:

```bash
pytest -v tests/
```

---

## 🧠 Model Architecture

- **Input Layer**: `(150, 150, 3)`
- **Block 1**: Conv2D(32, 3x3, ReLU) + MaxPooling2D(2x2)
- **Block 2**: Conv2D(64, 3x3, ReLU) + MaxPooling2D(2x2)
- **Block 3**: Conv2D(128, 3x3, ReLU) + MaxPooling2D(2x2)
- **Head**: Flatten -> Dense(128, ReLU) -> Dense(6, Softmax)
- **Format**: Native Keras 3 (`.keras`) with backwards-compatible `.h5` support

---

Created by Vincent Huynh | 📧 vintendohuynh@gmail.com
