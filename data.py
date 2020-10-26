import tensorflow as tf


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


def normalize(scan, min_value=-1000, max_value=400):
    """Normalize a CT scan with values in [0, 1].

    min_value is -1000 by default, which in HU scale is air.
    max_value is 400 by default, which in HU scale are bones.
    """
    scan = tf.clip_by_value(scan, min_value, max_value)
    scan = (scan - min_value) / (max_value - min_value)
    return scan


def add_channel_axis(scan):
    "Add the channel axis at the end."
    return tf.expand_dims(scan, axis=-1)


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
