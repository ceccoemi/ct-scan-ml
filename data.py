import numpy as np
import tensorflow as tf


def example_to_tensor(example, with_label=False):
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


def example_to_labeled_volume(example):
    volume_features = tf.io.parse_single_example(
        example,
        {
            "z": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.int64),
            "chn": tf.io.FixedLenFeature([], tf.int64),
            "volume_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
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
    label = volume_features["label"]
    label = tf.cast(label, tf.uint8)
    return volume, [label]


@tf.function
def normalize(t):
    "Normalize the input tensor with values in [0, 1]"
    normalized_t, _ = tf.linalg.normalize(t, ord=np.inf)
    return normalized_t


@tf.function
def normalize_labeled(t, label):
    "Normalize the input tensor with values in [0, 1]"
    normalized_t, _ = tf.linalg.normalize(t, ord=np.inf)
    return normalized_t, label


def tfrecord_dataset(tfrecord_fname):
    "Return a dataset from a tfrecord file with normalized data"
    return tf.data.TFRecordDataset(tfrecord_fname).map(
        example_to_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def tfrecord_labeled_dataset(tfrecord_fname):
    "Return a dataset with labels from a tfrecord file with normalized data"
    return tf.data.TFRecordDataset(tfrecord_fname).map(
        example_to_labeled_volume,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def ksplit_dataset(k, dataset, cardinality=None, seed=None):
    "Split a dataset into a list of k datasets"
    if not cardinality:
        cardinality = sum(1 for _ in dataset)
    assert 2 <= k <= cardinality
    dataset = dataset.shuffle(
        buffer_size=cardinality, reshuffle_each_iteration=False, seed=seed
    )
    split_size = cardinality // k
    remainder = cardinality % k
    splits = []
    for _ in range(k - 1):
        splits.append(dataset.take(split_size))
        dataset = dataset.skip(split_size)
    splits.append(dataset.take(split_size + remainder))
    return splits


def kfolds(k, dataset, cardinality=None, seed=None):
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


def train_test_split(dataset, test_perc=0.1, cardinality=None, seed=None):
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


def classification_dataset(
    small_neg_tfrecord,
    big_neg_tfrecord,
    small_pos_tfrecord,
    big_pos_tfrecord,
    return_size=False,
    seed=None,
):
    """
    Return a dataset used for classification,
    where each element is of the form:

        ((small_x, big_x), label)
    """
    neg_x = tf.data.Dataset.zip(
        (
            tfrecord_dataset(small_neg_tfrecord),
            tfrecord_dataset(big_neg_tfrecord),
        )
    )
    num_neg_samples = sum(1 for _ in neg_x)
    neg_dataset = tf.data.Dataset.zip(
        (
            neg_x,
            tf.data.Dataset.from_tensor_slices(np.int8([[0]])).repeat(
                num_neg_samples
            ),
        )
    )
    assert sum(1 for _ in neg_dataset) == num_neg_samples
    pos_x = tf.data.Dataset.zip(
        (
            tfrecord_dataset(small_pos_tfrecord),
            tfrecord_dataset(big_pos_tfrecord),
        )
    )
    num_pos_samples = sum(1 for _ in pos_x)
    pos_dataset = tf.data.Dataset.zip(
        (
            pos_x,
            tf.data.Dataset.from_tensor_slices(np.int8([[1]])).repeat(
                num_pos_samples
            ),
        )
    )
    assert sum(1 for _ in pos_dataset) == num_pos_samples
    total_samples = num_neg_samples + num_pos_samples
    dataset = neg_dataset.concatenate(pos_dataset).shuffle(
        buffer_size=total_samples, seed=seed, reshuffle_each_iteration=False
    )
    if return_size:
        return dataset, total_samples
    else:
        return dataset
