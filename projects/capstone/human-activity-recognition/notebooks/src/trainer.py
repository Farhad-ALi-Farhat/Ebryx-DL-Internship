import time

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from .visualization import plot_history
from .evaluation import evaluate_model


def train_model(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    epochs=100,
    batch_size=32,
):

    print(f"\nTraining {model_name}")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            f"../models/{model_name}.keras",
            save_best_only=True
        )
    ]

    start = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start

    plot_history(history, model_name)

    evaluation = evaluate_model(
        model,
        X_test,
        y_test,
        class_names,
        model_name,
    )
    
    result = {
        "Model": model_name,
        "Accuracy": evaluation["accuracy"],
        "Precision": evaluation["precision"],
        "Recall": evaluation["recall"],
        "F1 Score": evaluation["f1_score"],
        "Parameters": model.count_params(),
        "Epochs": len(history.history["loss"]),
        "Training Time (s)": round(training_time, 2),
    }
    
    return history, result
