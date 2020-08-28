import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import tensorflow as tf
from tqdm import tqdm
import nrrd
from pydicom import dcmread


def preprocess_scan(scan, downsample):
    "Apply some preprocessing to the image"
    if downsample != 1:
        scan = ndimage.zoom(scan, 1 / downsample)
    return scan


def scan_to_example(scan):
    "Convert a scan (a NumPy array) to an Example class"
    z, y, x = scan.shape
    scan_raw = scan.tostring()
    scan_features = {
        "z": tf.train.Feature(int64_list=tf.train.Int64List(value=[z])),
        "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        "x": tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
        "scan_raw": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[scan_raw])
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=scan_features))


def _split_into_subsequences(data, s):
    """
    Split the input sequence data into sublist of size s.

    >>> s = "abcdefg"
    >>> _split_into_subsequences(s, 2)
    ['ab', 'cd', 'ef', 'g']
    """
    return [data[x : x + s] for x in range(0, len(data), s)]


def convert_nrrd(path_glob, output_dir_name, downsample):
    nrrd_files = [str(f) for f in Path(".").glob(path_glob)]
    nrrd_files = _split_into_subsequences(nrrd_files, 10)
    output_dir = Path(output_dir_name)
    output_dir.mkdir()
    for i, chunk in tqdm(
        enumerate(nrrd_files, start=1), total=len(nrrd_files)
    ):
        tfrecord_fname = str(output_dir / f"{i:02}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_fname) as writer:
            for fname in chunk:
                scan, _ = nrrd.read(fname, index_order="C")
                scan = scan.astype(np.float32)
                scan = preprocess_scan(scan, downsample)
                example = scan_to_example(scan)
                writer.write(example.SerializeToString())


def convert_dicom(path_glob, output_dir_name, downsample):
    dcm_directories = list(Path(".").glob(path_glob))
    dcm_directories = _split_into_subsequences(dcm_directories, 10)
    output_dir = Path(output_dir_name)
    output_dir.mkdir()
    for i, chunk in tqdm(
        enumerate(dcm_directories, start=1), total=len(dcm_directories)
    ):
        tfrecord_fname = str(output_dir / f"{i:02}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_fname) as writer:
            for dcm_dir in chunk:
                dcm_files = Path(dcm_dir).glob("*.dcm")
                dcm_slices = [dcmread(str(f)) for f in dcm_files]
                is_volume = hasattr(dcm_slices[0], "SliceLocation")
                if is_volume:
                    dcm_slices = sorted(
                        dcm_slices, key=lambda x: x.SliceLocation
                    )
                    if all(
                        s.pixel_array.shape == (512, 512) for s in dcm_slices
                    ):
                        scan = np.stack([s.pixel_array for s in dcm_slices])
                        scan = scan.astype(np.float32)
                        scan = preprocess_scan(scan, downsample)
                        example = scan_to_example(scan)
                        writer.write(example.SerializeToString())


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    parser = argparse.ArgumentParser(
        description="Convert multiple CT scans to a single tfrecord file"
    )
    parser.add_argument("file_type", choices=["nrrd", "dicom"])
    parser.add_argument(
        "path_glob",
        help="Glob that identifies all the nrrd/dicom files to convert (must be inside quotes)",
    )
    parser.add_argument(
        "-d", "--downsample", type=float, default=1, help="Downscaling factor"
    )
    parser.add_argument(
        "output_dir",
        help="Name of the directory where to store the tfrecords files",
    )

    args = parser.parse_args()

    if args.file_type == "nrrd":
        convert_nrrd(args.path_glob, args.output_dir, args.downsample)
    if args.file_type == "dicom":
        convert_dicom(args.path_glob, args.output_dir, args.downsample)
