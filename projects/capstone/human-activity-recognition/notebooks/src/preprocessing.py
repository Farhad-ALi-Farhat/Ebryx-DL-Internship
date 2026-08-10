import numpy as np
from sklearn.preprocessing import OneHotEncoder

def encode_labels(y_train, y_test):
    """
    Convert labels from 1–6 to 0–5 and one-hot encode them.
    """
    y_train = y_train - 1
    y_test = y_test - 1

    encoder = OneHotEncoder(sparse_output=False)

    y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

    return (
        y_train_encoded,
        y_test_encoded,
    )