from pathlib import Path

import numpy as np

SIGNAL_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
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


def load_inertial_signals(dataset_path, split):
    """
    Load inertial signal data.

    Returns
    -------
    ndarray
        Shape: (samples, 128, 9)
    """

    dataset_path = Path(dataset_path)

    signal_path = dataset_path / split / "Inertial Signals"

    signals = []

    for signal in SIGNAL_FILES:
        file = signal_path / f"{signal}_{split}.txt"
        signals.append(np.loadtxt(file))

    return np.transpose(np.array(signals), (1, 2, 0))


def load_dataset(dataset_path):
    """
    Load train/test data and labels.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    dataset_path = Path(dataset_path)

    train_path = dataset_path / "train"
    test_path = dataset_path / "test"

    X_train = load_inertial_signals(dataset_path, "train")
    X_test = load_inertial_signals(dataset_path, "test")

    y_train = np.loadtxt(train_path / "y_train.txt").astype(int)
    y_test = np.loadtxt(test_path / "y_test.txt").astype(int)

    return X_train, X_test, y_train, y_test