import tensorflow as tf

from config import batch_size, input_shape, validation_size, test_size


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


def get_datasets(tfrecord_fnames):
    "Return training, validation and test set"
    dataset = tf.data.TFRecordDataset(tfrecord_fnames)
    dataset = dataset.map(example_to_tensor)
    dataset = dataset.map(normalize)
    dataset = dataset.map(
        lambda x: tf.expand_dims(x, axis=-1)  # add the channel dimension
    )

    test_dataset = dataset.take(test_size)
    test_dataset = test_dataset.batch(1)
    dataset = dataset.skip(test_size)
    dataset = dataset.padded_batch(
        batch_size=batch_size, padded_shapes=input_shape,
    )
    val_dataset = dataset.take(validation_size)
    train_dataset = dataset.skip(validation_size)
    train_dataset = train_dataset.shuffle(
        buffer_size=64, reshuffle_each_iteration=True
    )
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, val_dataset, test_dataset
