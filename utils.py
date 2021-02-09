from math import ceil, floor

import tensorflow as tf
import numpy as np
from pydicom import dcmread


def read_dcm(dcm_dir, reverse_z=True):
    """Read the DICOM slices and convert it to a single numpy array.

    dcm_dir is the directory where the dcm files of the slices are placed.
    Return a numpy array [z, y, x]
    """
    dcm_slices = [dcmread(f) for f in dcm_dir.glob("*.dcm")]
    dcm_slices = sorted(
        dcm_slices, key=lambda x: x.SliceLocation, reverse=reverse_z
    )
    scan = np.stack([s.pixel_array for s in dcm_slices])
    return scan


def extract_patch(volume, median_voxel, size):
    "Extract a 3D patch of the specified size from the input volume"
    z, y, x = volume.shape
    zloc, yloc, xloc = median_voxel
    zsize, ysize, xsize = size
    assert (
        xloc < x and yloc < y and zloc < z
    ), f"Can't find loc ({zloc}, {yloc}, {xloc}) with input shape {volume.shape}"
    return volume[
        (zloc - zsize // 2) : (zloc + zsize // 2),
        (yloc - ysize // 2) : (yloc + ysize // 2),
        (xloc - xsize // 2) : (xloc + xsize // 2),
    ]


def pad_to_shape(volume, shape):
    "Pad the input volume (3 dimensions) to have the input shape"
    assert (
        len(volume.shape) == 3
    ), f"Expected 3 dimensions, input has {len(volume.shape)} dimensions."
    assert (
        volume.shape <= shape
    ), f"Input volume (found {volume.shape}) can't be greater than shape (found {shape})."
    if volume.shape == shape:
        return volume
    dim1, dim2, dim3 = volume.shape
    target_dim1, target_dim2, target_dim3 = shape
    pad_dim1 = target_dim1 - dim1
    pad_dim2 = target_dim2 - dim2
    pad_dim3 = target_dim3 - dim3
    left_pad_dim1, right_pad_dim1 = ceil(pad_dim1 / 2), floor(pad_dim1 / 2)
    left_pad_dim2, right_pad_dim2 = ceil(pad_dim2 / 2), floor(pad_dim2 / 2)
    left_pad_dim3, right_pad_dim3 = ceil(pad_dim3 / 2), floor(pad_dim3 / 2)
    return np.pad(
        volume,
        (
            (left_pad_dim1, right_pad_dim1),
            (left_pad_dim2, right_pad_dim2),
            (left_pad_dim3, right_pad_dim3),
        ),
    )


def volume_to_example(volume):
    "Convert a volume (a NumPy array) to a tf.train.Example class"
    z, y, x, chn = volume.shape
    volume_raw = volume.tostring()
    volume_features = {
        "z": tf.train.Feature(int64_list=tf.train.Int64List(value=[z])),
        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
        "chn": tf.train.Feature(int64_list=tf.train.Int64List(value=[chn])),
        "volume_raw": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[volume_raw])
        ),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=volume_features)
    )


def volume_to_labeled_example(volume, label):
    "Convert a tuple (volume, label) to a tf.train.Example class"
    z, y, x, chn = volume.shape
    volume_raw = volume.tostring()
    volume_features = {
        "z": tf.train.Feature(int64_list=tf.train.Int64List(value=[z])),
        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
        "chn": tf.train.Feature(int64_list=tf.train.Int64List(value=[chn])),
        "volume_raw": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[volume_raw])
        ),
        "label": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])
        ),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=volume_features)
    )
