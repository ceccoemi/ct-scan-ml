import tensorflow as tf


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
