# Vehicle & Animal Image Classifier

A multi-class Convolutional Neural Network (CNN) built from scratch to classify images of vehicles and animals. Trained on the CIFAR-10 dataset using Python and TensorFlow.

## Overview

This project implements a deep CNN with three convolutional blocks, batch normalization, and progressive dropout regularization. The model is evaluated using standard classification metrics including precision, recall, F1 score, and confusion matrices.

## Model Architecture

- **Block 1:** 2x Conv2D (32 filters) → BatchNorm → MaxPool → Dropout (0.2)
- **Block 2:** 2x Conv2D (64 filters) → BatchNorm → MaxPool → Dropout (0.3)
- **Block 3:** 2x Conv2D (128 filters) → BatchNorm → MaxPool → Dropout (0.4)
- **Head:** Flatten → Dense (128) → BatchNorm → Dropout (0.5) → Softmax (10 classes)

## Classes

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

```bash
python cifar10_cnn.py
```

Running the script will automatically:
- Download and preprocess the CIFAR-10 dataset
- Train the model for 50 epochs
- Save `training_history.png` (accuracy/loss curves)
- Save `model_architecture.txt` (layer summary)
- Save `results.txt` (final test metrics)

## Output Files

| File | Description |
|------|-------------|
| `cifar10_cnn.py` | Main model training script |
| `training_history.png` | Accuracy and loss curves over epochs |
| `model_architecture.txt` | Model layer summary |
| `results.txt` | Final test loss and accuracy |

## Techniques Used

- Data augmentation and normalization
- Batch normalization for training stability
- Progressive dropout to prevent overfitting
- Train/validation/test splits
- Confusion matrix and F1 score evaluation
