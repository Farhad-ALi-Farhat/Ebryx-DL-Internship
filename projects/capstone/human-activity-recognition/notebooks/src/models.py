from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    SimpleRNN,
    LSTM,
    GRU,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D
)


def build_dense():

    model = Sequential([
        Input(shape=(128, 9)),

        Flatten(),

        Dense(512, activation="relu"),
        Dropout(0.3),

        Dense(256, activation="relu"),
        Dropout(0.3),

        Dense(128, activation="relu"),

        Dense(6, activation="softmax")
    ])

    return model

def build_rnn():

    model = Sequential([
        Input(shape=(128, 9)),

        SimpleRNN(
            128,
            activation="tanh"
        ),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dense(6, activation="softmax")
    ])

    return model

def build_lstm():

    model = Sequential([
        Input(shape=(128, 9)),

        LSTM(128),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dense(6, activation="softmax")
    ])

    return model

def build_gru():

    model = Sequential([
        Input(shape=(128, 9)),

        GRU(128),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dense(6, activation="softmax")
    ])

    return model

def build_cnn():

    model = Sequential([
        Input(shape=(128, 9)),

        Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        MaxPooling1D(pool_size=2),

        Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),

        GlobalAveragePooling1D(),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(6, activation="softmax")
    ])

    return model

def build_cnn_lstm():

    model = Sequential([
        Input(shape=(128, 9)),

        Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),

        MaxPooling1D(pool_size=2),

        LSTM(64),

        Dropout(0.3),

        Dense(64, activation="relu"),

        Dense(6, activation="softmax")
    ])

    return model