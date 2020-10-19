from pathlib import Path

import tensorflow as tf

from config import (
    seed,
    data_root_dir,
    tcia_glob,
    nrrd_glob,
    batch_size,
    input_shape,
    validation_num_samples,
    test_num_samples,
)


def example_to_tensor(example):
    "Reconstruct a 3D scan from a tf.train.Example."
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


def normalize(scan):
    "Normalize a CT scan with values in [0, 1]."
    min_value = -1000  # =< 1000 is air
    max_value = 400  # >= 400 is bones
    scan = tf.clip_by_value(scan, -1000, 400)
    scan = (scan - min_value) / (max_value - min_value)
    return scan


def add_channel_axis(scan):
    "Add the channel axis at the end."
    return tf.expand_dims(scan, axis=-1)


def train_test_split(dataset, test_perc=0.1, cardinality=None, seed=123):
    """Return a tuple (train_dataset, test_dataset).

    The dataset is shuffled with the specified seed.

    If cardinality is provided, the method will run faster,
    otherwise it has to determine the number of elements of
    the dataset.
    """
    if not cardinality:
        cardinality = sum(1 for _ in dataset)
    test_size = int(cardinality * test_perc)
    dataset = dataset.shuffle(
        buffer_size=cardinality, seed=seed, reshuffle_each_iteration=False
    )
    test_dataset = dataset.take(test_size)
    train_dataset = dataset.skip(test_size)
    return train_dataset, test_dataset


def get_datasets():
    "Return training, validation and test set"
    data_dir = Path(data_root_dir)
    assert data_dir.exists(), f"{data_dir} directory does not exists."
    tfrecord_fnames = [
        str(p)
        for g in (
            data_dir.glob(tcia_glob),
            data_dir.glob(nrrd_glob),
        )
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
    dataset = dataset.shuffle(
        buffer_size=32, seed=seed, reshuffle_each_iteration=False
    )
    test_dataset = dataset.take(test_num_samples)
    test_dataset = test_dataset.batch(1)
    dataset = dataset.skip(test_num_samples)
    val_dataset = dataset.take(validation_num_samples)
    val_dataset = val_dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=input_shape,
    )
    train_dataset = dataset.skip(validation_num_samples)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=input_shape,
    )
    train_dataset = train_dataset.cache()  # must be called before shuffle
    train_dataset = train_dataset.shuffle(
        buffer_size=64, reshuffle_each_iteration=True
    )
    # train_dataset = train_dataset.take(16)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset
