import tensorflow as tf


def example_to_tensor(example_proto, dtype):
    "Reconstruct a 3D scan from an instance in a tfrecords file"
    scan_features = tf.io.parse_single_example(
        example_proto,
        {
            "z": tf.io.FixedLenFeature([], tf.int64),
            "y": tf.io.FixedLenFeature([], tf.int64),
            "x": tf.io.FixedLenFeature([], tf.int64),
            "scan_raw": tf.io.FixedLenFeature([], tf.string),
        },
    )
    scan_1d = tf.io.decode_raw(scan_features["scan_raw"], dtype)
    scan = tf.reshape(
        scan_1d, (scan_features["z"], scan_features["y"], scan_features["x"])
    )
    scan = tf.expand_dims(scan, axis=-1)  # add the channel dimension
    max_value = tf.reduce_max(scan)
    min_value = tf.reduce_min(scan)
    scan = (scan - min_value) / (max_value - min_value)  # normalize in [0, 1]
    return scan
