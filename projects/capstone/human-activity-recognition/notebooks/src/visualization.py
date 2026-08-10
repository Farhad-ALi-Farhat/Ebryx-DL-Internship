import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def plot_history(history, model_name):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")

    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")

    axes[1].set_title(f"{model_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, model_name):

    fig, ax = plt.subplots(figsize=(7, 7))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    ).plot(ax=ax)

    plt.title(f"{model_name} Confusion Matrix")

    plt.show()