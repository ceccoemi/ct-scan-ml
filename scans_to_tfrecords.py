import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import tensorflow as tf
from tqdm import tqdm
import nrrd
from pydicom import dcmread
import nibabel as nib


def preprocess_scan(scan, downsample):
    "Apply some preprocessing to the image"
    if downsample != 1:
        scan = ndimage.zoom(scan, 1 / downsample)
    scan = scan.astype(np.float32)
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


def split_into_subsequences(data, s):
    """
    Split the input sequence into sublist of size s.

    >>> s = "abcdefg"
    >>> split_into_subsequences(s, 2)
    ['ab', 'cd', 'ef', 'g']
    """
    return [data[x : x + s] for x in range(0, len(data), s)]


def save_scan(writer, scan):
    """Serialize the scan in a tfrecord file.

    If the scan is not e regular volume (the first axis
    has length 0) then return doing nothing.
    """
    if scan.shape[0] < 1:
        return
    example = scan_to_example(scan)
    writer.write(example.SerializeToString())


def convert_nrrd(path_glob, output_dir_name, downsample):
    nrrd_files = [str(f) for f in Path(".").glob(path_glob)]
    nrrd_files = split_into_subsequences(nrrd_files, 10)
    output_dir = Path(output_dir_name)
    output_dir.mkdir()
    for i, chunk in tqdm(
        enumerate(nrrd_files, start=1), total=len(nrrd_files)
    ):
        tfrecord_fname = str(output_dir / f"{i:02}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_fname) as writer:
            for fname in chunk:
                scan, _ = nrrd.read(fname, index_order="C")
                scan = preprocess_scan(scan, downsample)
                save_scan(writer, scan)


def convert_dicom(path_glob, output_dir_name, downsample):
    dcm_directories = list(Path(".").glob(path_glob))
    dcm_directories = split_into_subsequences(dcm_directories, 10)
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
                        scan = preprocess_scan(scan, downsample)
                        save_scan(writer, scan)


def convert_nifti(path_glob, output_dir_name, downsample):
    nrrd_files = [str(f) for f in Path(".").glob(path_glob)]
    nrrd_files = split_into_subsequences(nrrd_files, 10)
    output_dir = Path(output_dir_name)
    output_dir.mkdir()
    for i, chunk in tqdm(
        enumerate(nrrd_files, start=1), total=len(nrrd_files)
    ):
        tfrecord_fname = str(output_dir / f"{i:02}.tfrecord")
        with tf.io.TFRecordWriter(tfrecord_fname) as writer:
            for fname in chunk:
                nib_obj = nib.load(fname)
                scan = nib_obj.get_fdata().T  # transpose to obtain C order
                scan = preprocess_scan(scan, downsample)
                save_scan(writer, scan)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    parser = argparse.ArgumentParser(
        description="Convert multiple CT scans to a single tfrecord file"
    )
    parser.add_argument("file_type", choices=["nrrd", "dicom", "nifti"])
    parser.add_argument(
        "path_glob",
        help="Glob that identifies all the nrrd/dicom/nifti files to convert (must be inside quotes)",
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
    if args.file_type == "nifti":
        convert_nifti(args.path_glob, args.output_dir, args.downsample)
