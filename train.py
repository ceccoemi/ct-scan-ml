from tensorflow import keras


def train_model(
    model,
    train_dataset,
    val_dataset,
    patience,
    monitor_metric,
    model_fname,
    log_dir,
    max_epochs=1000,
    verbose_training=0,
    verbose_checkpoint=0,
):
    """
    Train the model and return the best model found with early stopping.

    model_fname is the filename where the best model will be saved
    log_dir is the directory where to write the Tensorboard logs
    """
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=max_epochs,
        verbose=verbose_training,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                model_fname,
                monitor=monitor_metric,
                verbose=verbose_checkpoint,
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
            ),
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=False,
                profile_batch=0,
            ),
        ],
    )
    model = keras.models.load_model(model_fname)
    return model
