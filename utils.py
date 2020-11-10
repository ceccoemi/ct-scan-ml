import tensorflow as tf


def example_to_tensor(example):
    "Reconstruct a 3D volume from a tf.train.Example."
    volume_features = tf.io.parse_single_example(
        example,
        {
            "z": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.int64),
            "volume_raw": tf.io.FixedLenFeature([], tf.string),
        },
    )
    volume_1d = tf.io.decode_raw(volume_features["volume_raw"], tf.float32)
    volume = tf.reshape(
        volume_1d,
        (volume_features["z"], volume_features["y"], volume_features["x"]),
    )
    return volume


@tf.function
def add_channel_axis(image):
    "Append the channel axis to the input image (2D or 3D)."
    return tf.expand_dims(image, axis=-1)


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
