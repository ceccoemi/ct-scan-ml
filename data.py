import numpy as np
import tensorflow as tf

from config import SEED


def example_to_tensor(example):
    "Reconstruct a 3D volume from a tf.train.Example."
    volume_features = tf.io.parse_single_example(
        example,
        {
            "z": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.int64),
            "chn": tf.io.FixedLenFeature([], tf.int64),
            "volume_raw": tf.io.FixedLenFeature([], tf.string),
        },
    )
    volume_1d = tf.io.decode_raw(volume_features["volume_raw"], tf.float32)
    volume = tf.reshape(
        volume_1d,
        (
            volume_features["z"],
            volume_features["y"],
            volume_features["x"],
            volume_features["chn"],
        ),
    )
    return volume


@tf.function
def normalize(t):
    "Normalize the input tensor with values in [0, 1]"
    normalized_t, _ = tf.linalg.normalize(t, ord=np.inf)
    return normalized_t


def tfrecord_dataset(tfrecord_fname):
    "Return a dataset from a tfrecord file with normalized data"
    return (
        tf.data.TFRecordDataset(tfrecord_fname)
        .map(
            example_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        .map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )


def ksplit_dataset(k, dataset, cardinality=None, seed=SEED):
    "Split a dataset into k datasets and drop the remaining elements"
    if not cardinality:
        cardinality = sum(1 for _ in dataset)
    assert 2 <= k <= cardinality
    dataset = dataset.shuffle(
        buffer_size=cardinality, reshuffle_each_iteration=False, seed=seed
    )
    split_size = cardinality // k
    splits = []
    for _ in range(k):
        splits.append(dataset.take(split_size))
        dataset = dataset.skip(split_size)
    return splits


def kfolds(k, dataset, cardinality=None, seed=SEED):
    "Generator of training / test set with k fold"
    if not cardinality:
        cardinality = sum(1 for _ in dataset)
    folds = ksplit_dataset(k, dataset, cardinality, seed)
    for i, test_dataset in enumerate(folds):
        train_folds = [f for j, f in enumerate(folds) if j != i]
        train_dataset = train_folds[0]
        for d in train_folds[1:]:
            train_dataset = train_dataset.concatenate(d)
        yield train_dataset, test_dataset


def train_test_split(dataset, test_perc=0.1, cardinality=None, seed=SEED):
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
