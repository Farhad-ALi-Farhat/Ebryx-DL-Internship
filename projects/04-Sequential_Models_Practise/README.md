# Sequential Models Practice

This project provides a hands-on introduction to **Sequential Models** using **TensorFlow/Keras**. It demonstrates how recurrent neural networks process sequential data through practical experiments on synthetic datasets, covering both **time series forecasting** and **sequence classification**.

The notebook begins by generating a synthetic time series and training **LSTM** and **GRU** models to predict future values. It then explores sequence classification by recognizing different sequence patterns and concludes by visualizing the hidden representations learned by an LSTM using **Principal Component Analysis (PCA)**.

---

## Objectives

- Generate and visualize synthetic sequential data.
- Convert time series into supervised learning samples using sliding windows.
- Build and train an **LSTM** model for time series prediction.
- Build and compare a **GRU** model.
- Evaluate forecasting performance using regression metrics.
- Perform sequence pattern classification using an LSTM.
- Visualize learned hidden representations using PCA.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Project Workflow

```text
Generate Synthetic Time Series
            │
            ▼
Create Sliding Windows
            │
            ▼
Train/Test Split
            │
            ▼
Train LSTM Model
            │
            ▼
Evaluate & Visualize Predictions
            │
            ▼
Train GRU Model
            │
            ▼
Compare LSTM and GRU
            │
            ▼
Sequence Pattern Classification
            │
            ▼
Visualize Hidden States (PCA)
```

---

## Topics Covered

### Time Series Forecasting

- Synthetic time series generation
- Sliding window preprocessing
- LSTM regression
- GRU regression
- Model evaluation
- Prediction visualization

### Sequence Classification

- Synthetic sequence generation
- Multi-class sequence classification
- Classification report
- Confusion matrix

### Representation Learning

- Feature extraction from an LSTM layer
- Dimensionality reduction using PCA
- Hidden state visualization

---

## Learning Outcomes

After completing this notebook, you will be able to:

- Understand how sequential data is prepared for deep learning models.
- Apply sliding window techniques for sequence prediction.
- Build LSTM and GRU models using TensorFlow/Keras.
- Compare LSTM and GRU performance on a regression task.
- Train recurrent neural networks for sequence classification.
- Extract and visualize learned feature representations from hidden layers.

---

## Future Improvements

Possible extensions include:

- Forecasting real-world time series (e.g., stock prices or weather data).
- Multi-step sequence forecasting.
- Bidirectional LSTMs and GRUs.
- Attention mechanisms for sequence modeling.
- Transformer-based sequence models.

---

## Author

**Farhad Ali Farhat**
