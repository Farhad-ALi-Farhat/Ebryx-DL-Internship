# Binary Classification using Multi-Layer Perceptron (MLP)

## Overview

This project demonstrates how to build and evaluate a **Multi-Layer Perceptron (MLP)** for binary classification using the **Breast Cancer Wisconsin Dataset**. The project covers the complete deep learning workflow, from data preprocessing to model evaluation, while exploring the effects of different regularization techniques on model performance.

## Objectives

- Build a baseline MLP classifier using TensorFlow/Keras.
- Preprocess tabular data using train-test splitting and feature scaling.
- Visualize training and validation performance.
- Evaluate the model using multiple classification metrics.
- Apply Dropout and L2 Regularization to reduce overfitting.
- Compare the performance of different model variants.

## Dataset

- **Dataset:** Breast Cancer Wisconsin Dataset
- **Source:** Scikit-learn (`load_breast_cancer`)
- **Samples:** 569
- **Features:** 30 numerical features
- **Classes:**
  - Malignant (0)
  - Benign (1)

## Project Workflow

1. Import required libraries
2. Load and explore the dataset
3. Split the data into training and testing sets
4. Standardize features using `StandardScaler`
5. Build a baseline MLP model
6. Train the model
7. Visualize loss and accuracy curves
8. Evaluate model performance
9. Apply Dropout regularization
10. Apply L2 regularization
11. Combine Dropout and L2 regularization
12. Compare the results of all models

## Model Architecture

```
Input (30 Features)
        │
        ▼
Dense (64, ReLU)
        │
        ▼
Dense (32, ReLU)
        │
        ▼
Dense (1, Sigmoid)
```

## Regularization Techniques

The project compares four different models:

- Baseline MLP
- MLP with Dropout
- MLP with L2 Regularization
- MLP with Dropout + L2 Regularization

## Evaluation Metrics

The models are evaluated using:

- Test Loss
- Test Accuracy
- Classification Report
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

## Results

| Model | Test Accuracy | ROC-AUC |
|--------|--------------:|--------:|
| Baseline | 96.49% | 0.9931 |
| Dropout | 97.37% | 0.9938 |
| L2 Regularization | 97.37% | 0.9921 |
| Dropout + L2 | **98.25%** | **0.9948** |

The combined **Dropout + L2 Regularization** model achieved the best overall performance, demonstrating improved generalization on unseen data.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Key Concepts Covered

- Multi-Layer Perceptron (MLP)
- Binary Classification
- Feature Scaling
- Train-Test Split
- Dropout
- L2 Regularization (Weight Decay)
- Binary Crossentropy Loss
- Adam Optimizer
- Model Evaluation
- ROC-AUC Analysis

## Conclusion

This project demonstrated the complete workflow of building a neural network for tabular data classification. Starting with a baseline MLP, different regularization techniques were applied to improve model generalization. While the baseline model achieved strong performance, combining **Dropout** and **L2 Regularization** produced the highest test accuracy and ROC-AUC score, highlighting the effectiveness of regularization techniques in reducing overfitting and improving predictive performance.

---
**Part of the Ebryx Machine Learning Internship project series.**
