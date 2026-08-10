import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

CLASS_NAMES = [
    "Walking",
    "Walking Upstairs",
    "Walking Downstairs",
    "Sitting",
    "Standing",
    "Laying"
]

CHANNEL_NAMES = [
    "Body Acc X",
    "Body Acc Y",
    "Body Acc Z",
    "Body Gyro X",
    "Body Gyro Y",
    "Body Gyro Z",
    "Total Acc X",
    "Total Acc Y",
    "Total Acc Z",
]

MODEL_PATH = Path("../models/best_model.keras")
SAMPLE_PATH = Path("sample_data/sample.csv")


# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------

def validate_dataframe(df):

    if df.shape != (128, 9):
        st.error(
            f"Expected shape (128, 9), but received {df.shape}"
        )
        return False

    return True


def predict(df):

    X = df.values.astype(np.float32)

    X = np.expand_dims(X, axis=0)

    probs = model.predict(X, verbose=0)[0]

    pred = np.argmax(probs)

    confidence = probs[pred]

    return pred, confidence, probs


def plot_signals(df):

    fig, ax = plt.subplots(figsize=(14, 6))

    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    ax.set_title("Sensor Signals")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Sensor Reading")

    ax.legend(
        loc="upper right",
        fontsize=8,
        ncol=3
    )

    st.pyplot(fig)


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.title("🏃 HAR Project")

st.sidebar.markdown("""
### Human Activity Recognition

Deep Learning Capstone Project

**Dataset**

UCI HAR Dataset

**Best Model**

Automatically loaded from:

`models/best_model.keras`
""")

st.sidebar.success("Model Loaded Successfully")


# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("🏃 Human Activity Recognition")

st.write("""
Upload a **128 × 9 CSV** containing one sensor window.

Each row represents one timestep.

Each column represents one sensor channel.
""")


# -------------------------------------------------------
# Input Selection
# -------------------------------------------------------

option = st.radio(
    "Choose Input Source",
    [
        "Upload CSV",
        "Use Sample Data"
    ]
)

df = None

if option == "Upload CSV":

    uploaded = st.file_uploader(
        "Upload CSV",
        type="csv"
    )

    if uploaded:

        df = pd.read_csv(
            uploaded,
            header=None
        )

        df.columns = CHANNEL_NAMES

else:

    if SAMPLE_PATH.exists():

        df = pd.read_csv(
            SAMPLE_PATH,
            header=None
        )

        df.columns = CHANNEL_NAMES

    else:

        st.warning("Sample file not found.")


# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if df is not None:

    st.subheader("Data Preview")

    st.dataframe(df.head())

    if validate_dataframe(df):

        st.subheader("Sensor Signals")

        plot_signals(df)

        pred, confidence, probs = predict(df)

        st.subheader("Prediction")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Activity",
                CLASS_NAMES[pred]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        st.subheader("Class Probabilities")

        prob_df = pd.DataFrame({

            "Activity": CLASS_NAMES,

            "Probability": probs

        }).sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            prob_df.set_index("Activity")
        )

        st.dataframe(
            prob_df.style.format({
                "Probability": "{:.2%}"
            })
        )

        st.success("Inference completed successfully.")