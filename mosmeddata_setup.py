import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm

from utils import volume_to_example
from config import (
    CT_0_TFRECORD,
    CT_1_TFRECORD,
    CT_2_TFRECORD,
    CT_3_TFRECORD,
    CT_4_TFRECORD,
    COVID_SCAN_SHAPE,
)


def preprocess_scan(scan):
    z, y, x = scan.shape
    target_z, target_y, target_x, _ = COVID_SCAN_SHAPE
    scan = ndimage.zoom(
        scan, (target_z / z, target_y / y, target_x / x), order=5
    )
    scan = scan.astype(np.float32)
    scan = np.expand_dims(scan, axis=-1)  # add the channel dimension
    assert scan.shape == COVID_SCAN_SHAPE
    return scan


def main():
    parser = argparse.ArgumentParser(
        description="Convert MosMedData to 5 tfrecords",
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing all the NIFTI files downloaded from https://mosmed.ai/en/datasets/",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for category_dir in tqdm(data_dir.iterdir(), total=5):
        assert category_dir.name in ("CT-0", "CT-1", "CT-2", "CT-3", "CT-4")
        category_name = category_dir.name
        if category_name == "CT-0":
            tfrecord_fname = CT_0_TFRECORD
        elif category_name == "CT-1":
            tfrecord_fname = CT_1_TFRECORD
        elif category_name == "CT-2":
            tfrecord_fname = CT_2_TFRECORD
        elif category_name == "CT-3":
            tfrecord_fname = CT_3_TFRECORD
        elif category_name == "CT-4":
            tfrecord_fname = CT_4_TFRECORD
        with tf.io.TFRecordWriter(tfrecord_fname) as writer:
            nib_fnames = category_dir.glob("*.nii.gz")
            for nib_fname in nib_fnames:
                scan = nib.load(nib_fname).get_fdata().T # get traspose to obtain "C order"
                scan = preprocess_scan(scan)
                scan_example = volume_to_example(scan)
                writer.write(scan_example.SerializeToString())


if __name__ == "__main__":
    main()
