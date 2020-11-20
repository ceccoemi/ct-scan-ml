from tensorflow import keras
import numpy as np


def best_num_epochs(
    model,
    train_dataset,
    val_dataset,
    patience,
    monitor_metric,
    log_dir,
    metric_mode="max",
    max_epochs=1000,
    verbose_training=0,
    verbose_early_stopping=0,
):
    """
    Train the model and return number of epochs used
    to each the best model parameters with early stopping.

    log_dir is the directory where to write the Tensorboard logs.
    """
    metric_mode = metric_mode.lower()
    assert metric_mode in ("max", "min"), "metric_mode must be 'max' or 'min'"
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        verbose=verbose_training,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
                verbose=verbose_early_stopping,
            ),
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=False,
                profile_batch=0,
            ),
        ],
    )
    if metric_mode == "max":
        return np.argmax(history.history[monitor_metric])
    elif metric_mode == "min":
        return np.argmin(history.history[monitor_metric])


def train_model(
    model,
    train_dataset,
    num_epochs,
    model_fname,
    verbose_training=0,
):
    """
    Train the model and return the final trained model.

    model_fname is the filename where the final model will be saved.
    """
    model.fit(
        train_dataset,
        epochs=num_epochs,
        verbose=verbose_training,
    )
    model.save(model_fname)
    return model
