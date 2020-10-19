import datetime
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tqdm import tqdm

from data import get_datasets
from model import build_autoencoder
from config import (
    use_mixed_precision,
    verbose_training,
    seed,
    epochs,
    learning_rate,
    patience,
    batch_size,
)


def train_func(
    model,
    optimizer,
    loss,
    train_dataset,
    val_dataset,
    model_dir,
    logs_dir,
    monitor_metric,
    metrics=None,
):
    """Perform a training and return the trained model
    (loaded at the best epoch).

    model_dir is the path of the directory where the model will be saved.
    logs_dir is the path of the tensorboard directory.
    monitor_metric is the metric to monitor for early stopping.
    metrics is a list of metrics to track during the training.
    """
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    best_checkpoint = f"{model_dir}/{start_time}.h5"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        best_checkpoint, save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor=monitor_metric, patience=patience
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=f"{logs_dir}/{start_time}",
        histogram_freq=1,
        write_graph=False,
        profile_batch=0,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )
    model = keras.models.load_model(best_checkpoint)
    return model


class EarlyStopping:
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_value = np.inf
        self.not_improving_epochs = 0
        self.early_stop = False

    def update(self, value):
        if value + self.delta >= self.best_value:
            if self.not_improving_epochs == self.patience:
                self.early_stop = True
            else:
                self.not_improving_epochs += 1
        else:
            self.not_improving_epochs = 0
            self.best_value = value

    def __call__(self, value):
        self.update(value)


def train(model, loss, optimizer, train_dataset, val_dataset):
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/{start_time_str}/"
    model_dir = f"models/{start_time_str}/"
    ckpt_dir = model_dir + "best_epoch_ckpt"
    writer = tf.summary.create_file_writer(log_dir)

    early_stopping = EarlyStopping(patience)

    for epoch in tqdm(range(epochs), disable=(not verbose_training)):

        ### TRAIN ###

        train_loss_metric = tf.keras.metrics.Mean(
            "train_loss", dtype=tf.float32
        )
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(batch)
                loss_value = loss(predictions, batch)
                if use_mixed_precision:
                    loss_value = optimizer.get_scaled_loss(loss_value)
            gradients = tape.gradient(loss_value, model.trainable_variables)
            if use_mixed_precision:
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            train_loss_metric.update_state(loss_value)
            with writer.as_default():
                for grad, param in zip(gradients, model.trainable_variables):
                    tf.summary.histogram(param.name, param, step=epoch)
                    tf.summary.histogram(
                        param.name + "/grad", grad, step=epoch
                    )

        train_loss_mean = train_loss_metric.result()
        with writer.as_default():
            tf.summary.scalar("Training loss", train_loss_mean, step=epoch)
        train_loss_metric.reset_states()

        ### VALIDATION ###

        val_loss_metric = tf.keras.metrics.Mean("val_loss", dtype=tf.float32)
        for batch in val_dataset:
            predictions = model(batch, training=False)
            val_loss_metric.update_state(loss(predictions, batch))

        val_loss_mean = val_loss_metric.result()
        with writer.as_default():
            tf.summary.scalar("Validation loss", val_loss_mean, step=epoch)
        val_loss_metric.reset_states()

        if verbose_training:
            print()
            print(f"Epoch : {epoch}")
            print(f"Training loss: {train_loss_mean}")
            print(f"Validation loss: {val_loss_mean}")

        ### EARLY STOPPING ###

        early_stopping.update(val_loss_mean)
        if early_stopping.early_stop:
            model.load_weights(ckpt_dir)
            break
        elif early_stopping.not_improving_epochs == 0:
            model.save_weights(ckpt_dir)

    model.save(model_dir)

    end_time = datetime.datetime.now()
    training_time = str(end_time - start_time).split(".")[0]

    with writer.as_default():
        tf.summary.text(
            "Hyperparameters",
            f"batch size = {batch_size}; "
            f"patience = {patience}; "
            f"learning rate = {learning_rate}; "
            f"seed = {seed}; "
            f"training time = {training_time}",
            step=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu", default="0", help="GPU id where to run the process"
    )
    args = parser.parse_args()

    with tf.device(f"/device:GPU:{args.gpu}"):
        train_dataset, val_dataset, _ = get_datasets()
        model = build_autoencoder()
        loss = keras.losses.MeanSquaredError()
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        if use_mixed_precision:
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer, loss_scale="dynamic"
            )
        train(model, loss, optimizer, train_dataset, val_dataset)
