from pathlib import Path

import tensorflow as tf

from config import (
    tcia_glob,
    nrrd_glob,
    batch_size,
    input_shape,
    validation_num_samples,
    test_num_samples,
)


def example_to_tensor(example):
    "Reconstruct a 3D scan from an example"
    scan_features = tf.io.parse_single_example(
        example,
        {
            "z": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.int64),
            "scan_raw": tf.io.FixedLenFeature([], tf.string),
        },
    )
    scan_1d = tf.io.decode_raw(scan_features["scan_raw"], tf.float32)
    scan = tf.reshape(
        scan_1d, (scan_features["z"], scan_features["y"], scan_features["x"])
    )
    return scan


def normalize(t):
    "Normalize the input tensor in [0, 1]"
    max_value = tf.reduce_max(t)
    min_value = tf.reduce_min(t)
    return (t - min_value) / (max_value - min_value)


def get_datasets():
    "Return training, validation and test set"
    data_dir = Path("data")
    tfrecord_fnames = [
        str(p)
        for g in (data_dir.glob(tcia_glob), data_dir.glob(nrrd_glob),)
        for p in g
    ]

    dataset = tf.data.TFRecordDataset(tfrecord_fnames)
    dataset = dataset.map(
        example_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda x: tf.expand_dims(x, axis=-1),  # add the channel axis
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.shuffle(buffer_size=32)
    test_dataset = dataset.take(test_num_samples)
    test_dataset = test_dataset.batch(1)
    dataset = dataset.skip(test_num_samples)
    val_dataset = dataset.take(validation_num_samples)
    val_dataset = val_dataset.padded_batch(
        batch_size=batch_size, padded_shapes=input_shape,
    )
    train_dataset = dataset.skip(validation_num_samples)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padded_shapes=input_shape,
    )
    train_dataset = train_dataset.cache()  # must be called before shuffle
    train_dataset = train_dataset.shuffle(
        buffer_size=64, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset
