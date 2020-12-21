import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import nibabel as nib
import tensorflow as tf
from tqdm import tqdm

from utils import volume_to_example
from config import (
    COVID_NEG_TFRECORD,
    COVID_POS_TFRECORD,
    RESIZED_SCAN_SHAPE,
    SCAN_SHAPE,
)


def preprocess_scan(scan):
    z, y, x = scan.shape
    target_z, target_y, target_x, _ = RESIZED_SCAN_SHAPE
    scan = ndimage.zoom(
        scan, (target_z / z, target_y / y, target_x / x), order=5
    )
    scan = scan.astype(np.float32)
    scan = scan[:, 32:-32, 16:-16]
    scan = np.expand_dims(scan, axis=-1)  # add the channel dimension
    assert scan.shape == SCAN_SHAPE
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
    with tf.io.TFRecordWriter(
        COVID_NEG_TFRECORD
    ) as neg_writer, tf.io.TFRecordWriter(COVID_POS_TFRECORD) as pos_writer:
        for category_dir in tqdm(data_dir.iterdir(), total=5):
            assert category_dir.name in (
                "CT-0",
                "CT-1",
                "CT-2",
                "CT-3",
                "CT-4",
            )
            category_name = category_dir.name
            if category_name == "CT-0":
                writer = neg_writer
            elif category_name in ("CT-1", "CT-2", "CT-3", "CT-4"):
                writer = pos_writer
            else:
                raise RuntimeError(
                    f"{category_name} is an unknown category in MosMedData"
                )

            nib_fnames = category_dir.glob("*.nii.gz")
            for nib_fname in nib_fnames:
                scan = (
                    nib.load(nib_fname).get_fdata().T
                )  # get traspose to obtain "C order"
                scan = preprocess_scan(scan)
                scan_example = volume_to_example(scan)
                writer.write(scan_example.SerializeToString())


if __name__ == "__main__":
    main()
