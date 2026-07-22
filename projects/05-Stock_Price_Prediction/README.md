# Stock Price Prediction using LSTM and GRU

This project demonstrates how **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks can be applied to **time series forecasting** by predicting future stock prices from historical market data.

Using **Apple (AAPL)** stock data downloaded from Yahoo Finance, the project follows a complete deep learning workflow—from data acquisition and preprocessing to model training, evaluation, and comparison. Both recurrent architectures are trained under identical conditions to provide a fair comparison of their forecasting performance.

---

## Project Objectives

- Download historical stock market data using Yahoo Finance.
- Perform exploratory data analysis (EDA).
- Prepare multivariate time-series data using a sliding window approach.
- Build and train an LSTM model.
- Build and train a GRU model.
- Prevent overfitting using EarlyStopping.
- Compare both models using multiple regression metrics.
- Visualize and interpret prediction results.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- yfinance

---

## Dataset

- **Source:** Yahoo Finance
- **Ticker:** AAPL (Apple Inc.)
- **Time Period:** January 2015 – January 2025

### Features

- Open
- High
- Low
- Close
- Volume

### Target

- Next day's **Closing Price**

---

## Project Workflow

```text
Download Historical Stock Data
            │
            ▼
Exploratory Data Analysis
            │
            ▼
Feature Selection
            │
            ▼
Min-Max Feature Scaling
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
Evaluate LSTM
            │
            ▼
Train GRU Model
            │
            ▼
Evaluate GRU
            │
            ▼
Compare Models
            │
            ▼
Visualize Predictions
```

---

## Model Architecture

Both models share the same architecture to ensure a fair comparison.

```text
Input (60 Days × 5 Features)
            │
            ▼
64 Recurrent Units
            │
            ▼
Dropout (0.2)
            │
            ▼
32 Recurrent Units
            │
            ▼
Dropout (0.2)
            │
            ▼
Dense (16, ReLU)
            │
            ▼
Dense (1)
```

The only difference between the models is the recurrent layer:

- **LSTM** for the first model.
- **GRU** for the second model.

---

## Training Strategy

To improve generalization and reduce overfitting, both models were trained using **EarlyStopping** with:

- Validation Loss Monitoring
- Patience = 5
- Restore Best Weights = True

This automatically restores the weights from the epoch with the lowest validation loss.

---

## Evaluation Metrics

The models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R²)

---

## Visualizations

The notebook includes:

- Closing Price Over Time
- Trading Volume Over Time
- Feature Distributions
- Correlation Matrix
- Training vs Validation Loss
- Actual vs Predicted Prices (LSTM)
- Actual vs Predicted Prices (GRU)
- LSTM vs GRU Comparison Plot

---

## Key Learning Outcomes

After completing this project, you will understand how to:

- Work with real-world financial time-series data.
- Perform exploratory data analysis for sequential datasets.
- Normalize multivariate features for deep learning.
- Convert sequential data into supervised learning samples using sliding windows.
- Build stacked LSTM and GRU networks.
- Apply EarlyStopping to reduce overfitting.
- Evaluate regression models using multiple performance metrics.
- Compare different recurrent neural network architectures on the same forecasting task.

---

## Results

Both recurrent models successfully learned temporal patterns from historical stock prices.

The introduction of **EarlyStopping** significantly improved generalization by preventing unnecessary training after the validation loss stopped improving. Among the two architectures, the **GRU model achieved better overall predictive performance** on the test dataset, demonstrating lower prediction error and a higher coefficient of determination (R²) than the LSTM model.

---

## Future Improvements

Possible extensions of this project include:

- Forecasting multiple future trading days.
- Predicting returns instead of raw prices.
- Adding technical indicators (RSI, MACD, Bollinger Bands, Moving Averages).
- Hyperparameter tuning using Keras Tuner or Optuna.
- Experimenting with Bidirectional LSTMs and GRUs.
- Comparing recurrent models with Transformer-based architectures.

---

## License

This project was completed as part of the **Ebryx Machine Learning Internship** for educational and learning purposes.
