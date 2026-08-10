# Human Activity Recognition using Deep Learning

A deep learning capstone project for recognizing human physical activities from smartphone inertial sensor data.

This project compares six different deep learning architectures — Dense Neural Network, Simple RNN, LSTM, GRU, 1D CNN, and CNN-LSTM — using the **UCI Human Activity Recognition Using Smartphones Dataset**.

The best-performing model, an **LSTM**, is integrated into a Streamlit application for interactive inference on unseen sensor windows.

---

## Project Overview

Human Activity Recognition (HAR) is the task of automatically identifying physical activities from sensor data.

This project uses accelerometer and gyroscope signals collected from a smartphone worn at the waist to classify six different activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

The project compares different neural network architectures to investigate how effectively they model temporal patterns in human movement.

The complete workflow is:

```text
UCI HAR Dataset
       │
       ▼
Data Loading
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Preprocessing
       │
       ▼
Model Training
       │
       ├── Dense
       ├── Simple RNN
       ├── LSTM
       ├── GRU
       ├── CNN
       └── CNN-LSTM
       │
       ▼
Model Evaluation
       │
       ▼
Model Comparison
       │
       ▼
Best Model Selection
       │
       ▼
Streamlit Application
       │
       ▼
Activity Prediction
```

---

## Objectives

The main objectives of this project are:

- Build a deep learning pipeline for human activity recognition.
- Work with multivariate time-series sensor data.
- Compare feed-forward, recurrent, convolutional, and hybrid architectures.
- Understand how different architectures handle sequential sensor data.
- Evaluate models using multiple classification metrics.
- Compare models based on both performance and parameter count.
- Identify the best-performing architecture.
- Save the trained model for inference.
- Build a simple Streamlit interface for real-world-style inference.

---

## Dataset

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset contains recordings from 30 subjects performing six different physical activities while carrying a smartphone on their waist.

The sensor measurements include accelerometer and gyroscope signals.

The data is organized into fixed-width windows of 128 readings.

### Activities

| Label | Activity |
|------:|----------|
| 1 | Walking |
| 2 | Walking Upstairs |
| 3 | Walking Downstairs |
| 4 | Sitting |
| 5 | Standing |
| 6 | Laying |

### Input Representation

For this project, the sensor data was organized into:

```text
(samples, 128 timesteps, 9 channels)
```

The nine input channels are:

1. Body Acc X
2. Body Acc Y
3. Body Acc Z
4. Body Gyro X
5. Body Gyro Y
6. Body Gyro Z
7. Total Acc X
8. Total Acc Y
9. Total Acc Z

The resulting datasets are:

```text
Training Shape : (7352, 128, 9)
Testing Shape  : (2947, 128, 9)
```

The original subject-based train/test split provided by the dataset was retained.

---

## Exploratory Data Analysis

The dataset was inspected before model training.

The analysis included:

- Dataset shape inspection
- Activity label inspection
- Class distribution
- Missing-value checking
- Sensor signal visualization
- Per-channel statistics
- Subject distribution
- Train/test subject separation

No missing values were found in the sensor data.

### Class Distribution

| Activity | Samples |
|----------|--------:|
| Laying | 1407 |
| Standing | 1374 |
| Sitting | 1286 |
| Walking | 1226 |
| Walking Upstairs | 1073 |
| Walking Downstairs | 986 |

The class distribution is reasonably balanced, although the number of samples varies between activities.

Sensor windows were also visualized across their 128 timesteps to examine temporal patterns in the different sensor channels.

---

## Deep Learning Models

Six architectures were implemented and trained using the same dataset and evaluation procedure.

The purpose was to compare different approaches to modeling the temporal structure of human activity sensor data.

### 1. Dense Neural Network

A fully connected neural network was used as a baseline.

The input sensor window is flattened before being passed through dense layers.

Because the model does not explicitly model temporal relationships, it provides a useful baseline for comparison with sequence-based architectures.

### 2. Simple RNN

A Simple Recurrent Neural Network was used to process the sensor sequence timestep by timestep.

The recurrent structure allows information from previous timesteps to influence later predictions.

This provides a basic recurrent baseline.

### 3. LSTM

Long Short-Term Memory networks are designed to model dependencies in sequential data.

The LSTM uses gating mechanisms to control the flow of information through the sequence.

It achieved the best overall performance in this experiment.

### 4. GRU

Gated Recurrent Units provide a gated recurrent architecture similar to LSTM while using a simpler internal structure.

The GRU achieved the second-best performance while using fewer parameters than the LSTM.

### 5. 1D CNN

A one-dimensional convolutional neural network was used to extract local temporal patterns from the sensor signals.

The convolutional layers can identify useful patterns across neighboring timesteps while keeping the model relatively compact.

### 6. CNN-LSTM

The CNN-LSTM combines convolutional feature extraction with recurrent sequence modeling.

```text
CNN
 ↓
Temporal Feature Extraction
 ↓
LSTM
 ↓
Sequence Modeling
 ↓
Classification
```

The architecture was designed to combine local temporal feature extraction with recurrent sequence modeling.

---

## Model Comparison

All six architectures were evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1 Score | Parameters | Epochs | Training Time (s) |
|-------|---------:|----------:|-------:|---------:|-----------:|-------:|------------------:|
| **LSTM** | **91.86%** | **92.07%** | **91.86%** | **91.73%** | 79,302 | 29 | 181.12 |
| **GRU** | 90.43% | 90.63% | 90.43% | 90.43% | 62,022 | 20 | 118.18 |
| **Dense** | 89.01% | 89.56% | 89.01% | 88.95% | 755,334 | 14 | 18.16 |
| **CNN** | 88.60% | 88.99% | 88.60% | 88.63% | 36,294 | 12 | 25.93 |
| **CNN-LSTM** | 88.33% | 88.54% | 88.33% | 88.33% | 40,518 | 13 | 65.73 |
| **Simple RNN** | 72.72% | 76.32% | 72.72% | 72.28% | 26,310 | 26 | 115.10 |

---

## Best Model

The **LSTM** achieved the highest test accuracy among the evaluated architectures.

```text
Test Accuracy : 91.86%
Precision      : 92.07%
Recall         : 91.86%
F1 Score       : 91.73%
Parameters     : 79,302
```

The final trained LSTM model is saved as:

```text
models/best_model.keras
```

---

## Model Analysis

### LSTM

The LSTM achieved the best overall performance.

Its results indicate that explicitly modeling temporal dependencies is beneficial for this human activity recognition task because the order and evolution of sensor measurements contain useful information about the activity being performed.

### GRU

The GRU achieved:

```text
Accuracy: 90.43%
```

while using fewer parameters than the LSTM:

```text
LSTM : 79,302 parameters
GRU  : 62,022 parameters
```

This makes GRU a useful alternative when a smaller recurrent model is preferred.

### Dense Network

The Dense model achieved:

```text
Accuracy: 89.01%
```

Despite not explicitly modeling temporal dependencies, it performed competitively.

However, it used:

```text
755,334 parameters
```

which is substantially more than the recurrent and convolutional models.

### CNN

The CNN achieved:

```text
Accuracy: 88.60%
Parameters: 36,294
```

This demonstrates that convolutional layers can extract useful local temporal patterns while maintaining a relatively small parameter count.

### CNN-LSTM

The CNN-LSTM achieved:

```text
Accuracy: 88.33%
```

Although the hybrid architecture combines convolutional and recurrent layers, it did not outperform the standalone LSTM in this experiment.

### Simple RNN

The Simple RNN achieved:

```text
Accuracy: 72.72%
```

This was substantially lower than both the LSTM and GRU.

The result demonstrates the practical advantage of gated recurrent architectures for this sequence classification problem.

---

## Error Analysis

The confusion matrices showed that one of the main sources of classification error occurs between:

```text
Sitting ↔ Standing
```

These two stationary activities can produce relatively similar sensor patterns, making them more difficult to distinguish.

This behavior was also observed during inference through the Streamlit application, where some genuine Sitting samples were classified as Standing.

This is treated as a model limitation rather than being manually corrected in the application.

The application reports the actual prediction produced by the trained model.

---

## Streamlit Application

The best-performing LSTM model was integrated into a Streamlit application for interactive inference.

The application allows a user to provide a sensor window and obtain an activity prediction.

### Application Features

- Upload a CSV sensor window.
- Validate the input dimensions.
- Preview the uploaded sensor data.
- Visualize the nine sensor channels.
- Run model inference.
- Display the predicted activity.
- Display prediction confidence.
- Display class probabilities.
- Test the model using sample UCI HAR test windows.

---

## Input Format

The application expects one sensor window containing:

```text
128 rows × 9 columns
```

Each row represents a timestep.

Each column represents a sensor channel.

The expected channels are:

```text
Body Acc X
Body Acc Y
Body Acc Z
Body Gyro X
Body Gyro Y
Body Gyro Z
Total Acc X
Total Acc Y
Total Acc Z
```

Sample CSV files used for testing are stored in:

```text
app/sample_data/
```

Example:

```text
app/sample_data/
├── walking.csv
├── walking_upstairs.csv
├── walking_downstairs.csv
├── sitting.csv
├── standing.csv
└── laying.csv
```

These samples are taken from the UCI HAR test data and represent unseen test windows.

---

## Project Structure

This project is part of the larger Deep Learning repository rather than a standalone Git repository.

The project is located at:

```text
projects/
└── capstone/
    └── human-activity-recognition/
```

The project structure is:

```text
human-activity-recognition/
│
├── app/
│   ├── streamlit_app.py
│   │
│   └── sample_data/
│       ├── walking.csv
│       ├── walking_upstairs.csv
│       ├── walking_downstairs.csv
│       ├── sitting.csv
│       ├── standing.csv
│       └── laying.csv
│
├── data/
│   └── UCI HAR Dataset/
│
├── models/
│   └── best_model.keras
│
├── notebooks/
│   ├── HAR_Deep_Learning.ipynb
│   │
│   └── src/
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── models.py
│       ├── trainer.py
│       ├── evaluation.py
│       ├── visualization.py
│       └── utils.py
│
├── results/
│   └── model_comparison.csv
│
├── requirements.txt
└── README.md
```

---

## Modular Code Structure

Helper functionality used during model development was moved into the `notebooks/src/` directory to keep the main notebook clean.

### `data_loader.py`

Responsible for loading the UCI HAR sensor data and associated labels.

### `preprocessing.py`

Contains preprocessing and label preparation functionality.

### `models.py`

Contains the different model architecture definitions used in the comparison.

### `trainer.py`

Contains the common training workflow used by the different models, including:

- Model compilation
- Training
- Early stopping
- Model checkpointing
- Training time measurement
- Metric collection

### `evaluation.py`

Contains model evaluation functionality, including:

- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- Confusion matrices

### `visualization.py`

Contains visualization utilities such as:

- Training history plots
- Sensor signal plots
- Confusion matrix visualization

### `utils.py`

Contains general helper utilities shared across the project.

---

## Model Saving

The trained models are saved using Keras model checkpoints.

The final selected LSTM model is stored as:

```text
models/best_model.keras
```

The Streamlit application loads this model for inference.

---

## Requirements

The project uses:

- Python 3.11
- TensorFlow 2.19
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Jupyter Lab

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit Application

From the `human-activity-recognition` directory:

```bash
streamlit run app/streamlit_app.py
```

The application loads:

```text
models/best_model.keras
```

and uses it to perform inference on uploaded or sample sensor windows.

---

## Running the Notebook

Launch Jupyter Lab:

```bash
jupyter lab
```

Open:

```text
notebooks/HAR_Deep_Learning.ipynb
```

The notebook contains the complete model development workflow:

1. Dataset loading
2. Exploratory data analysis
3. Data preprocessing
4. Model construction
5. Model training
6. Evaluation
7. Model comparison
8. Best-model selection

The helper modules used by the notebook are located under:

```text
notebooks/src/
```

---

## Technologies

- **Python**
- **TensorFlow**
- **Keras**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Streamlit**
- **Jupyter Lab**

---

## Deep Learning Concepts Demonstrated

This capstone brings together the major deep learning concepts covered during the internship up to recurrent neural networks:

- Neural networks
- Dense layers
- Time-series classification
- 1D Convolutional Neural Networks
- Recurrent Neural Networks
- Long Short-Term Memory networks
- Gated Recurrent Units
- CNN-RNN hybrid architectures
- Sequence modeling
- Multi-class classification
- Early stopping
- Model checkpointing
- Training and validation monitoring
- Confusion matrices
- Accuracy
- Precision
- Recall
- F1-score
- Model comparison
- Model deployment
- Inference on unseen data

---

## Future Improvements

Potential future extensions include:

- Hyperparameter optimization
- Sensor-data augmentation
- More extensive per-class error analysis
- Attention mechanisms
- Transformer-based time-series models
- Real-time sensor streaming
- Mobile deployment
- Cloud deployment
- Testing on additional human activity recognition datasets
- Incorporating additional sensor modalities

---

## Dataset Reference

**Human Activity Recognition Using Smartphones Dataset**

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L.

*A Public Domain Dataset for Human Activity Recognition Using Smartphones.*

European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning.

UCI Machine Learning Repository.

Dataset DOI:

```text
10.24432/C54S4K
```

---

## Author

**Farhad Ali**

Deep Learning Internship — Capstone Project
