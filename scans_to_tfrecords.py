import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import tensorflow as tf
from tqdm import tqdm
import nrrd
from pydicom import dcmread


def preprocess_scan(scan, downsample, dtype):
    "Apply some preprocessing to the images"
    if downsample != 1:
        scan = ndimage.zoom(scan, 1 / downsample)
    scan = scan.astype(dtype)
    return scan


def scan_example(scan):
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


def convert_nrrd(path_glob, downscale, dtype, tfrecords_fname):
    fnames = [str(f) for f in Path(".").glob(path_glob)]
    with tf.io.TFRecordWriter(tfrecords_fname) as writer:
        for fname in tqdm(fnames):
            scan, _ = nrrd.read(fname, index_order="C")
            scan = preprocess_scan(scan, downscale, dtype)
            example = scan_example(scan)
            writer.write(example.SerializeToString())


def convert_dicom(path_glob, downscale, dtype, tfrecord_fname):
    datadir = Path(path_glob)
    dcm_directories = list(datadir.glob("*/*/*/"))
    with tf.io.TFRecordWriter(tfrecord_fname) as writer:
        for dcm_dir in tqdm(dcm_directories):
            dcm_files = Path(dcm_dir).glob("*.dcm")
            dcm_slices = [dcmread(str(f)) for f in dcm_files]
            is_volume = len(dcm_slices) >= downscale and hasattr(
                dcm_slices[0], "SliceLocation"
            )
            if is_volume:
                dcm_slices = sorted(dcm_slices, key=lambda x: x.SliceLocation)
                if all(s.pixel_array.shape == (512, 512) for s in dcm_slices):
                    scan = np.stack([s.pixel_array for s in dcm_slices])
                    scan = preprocess_scan(scan, downscale, dtype)
                    example = scan_example(scan)
                    writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multiple CT scans to a single tfrecord file"
    )
    parser.add_argument("file_type", choices=["nrrd", "dicom"])
    parser.add_argument(
        "path_glob",
        help="Glob that identifies all the nrrd files to convert (must be inside quotes)",
    )
    parser.add_argument(
        "-d", "--downscale", type=float, default=1, help="Downscaling factor"
    )
    parser.add_argument("-t", "--dtype", default="float32", help="Data type")
    parser.add_argument(
        "tfrecord_fname", help="File name where to store the tfrecords",
    )

    args = parser.parse_args()

    if args.file_type == "nrrd":
        convert_nrrd(
            args.path_glob, args.downscale, args.dtype, args.tfrecord_fname
        )
    if args.file_type == "dicom":
        convert_dicom(
            args.path_glob, args.downscale, args.dtype, args.tfrecord_fname
        )
