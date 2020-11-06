import argparse
from pathlib import Path
import random
from itertools import cycle, islice

import tensorflow as tf
import numpy as np
from scipy import ndimage
import nibabel as nib
from tqdm import tqdm

from config import TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, CT_0, CT_1, CT_2, CT_3, SEED, SCAN_SHAPE


def scan_to_example(scan):
    "Convert a scan (a NumPy array) to an tf.train.Example class"
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


def resize_scan(scan):
    "Resize the input scan to SCAN_SHAPE and convert it to float32."
    z, y, x = scan.shape
    target_z, target_y, target_x = SCAN_SHAPE
    scan = ndimage.zoom(
        scan, (target_z / z, target_y / y, target_x / x), order=5
    )
    scan = scan.astype(np.float32)
    return scan


def convert_to_tfrecord_files(nifti_fnames, tfrecord_fname):
    "Write the nifti files to a single tfrecord file."
    with tf.io.TFRecordWriter(tfrecord_fname) as writer:
        for fname in tqdm(nifti_fnames):
            nib_obj = nib.load(fname)
            scan = nib_obj.get_fdata().T  # transpose to obtain C order
            scan = resize_scan(scan)
            example = scan_to_example(scan)
            writer.write(example.SerializeToString())


def main():
    parser = argparse.ArgumentParser(
        description="Split training (with resampling), validation and test for the MosMed dataset."
    )
    parser.add_argument(
        "data_input_root_dir",
        help="Directory that contains the CT-[0-4] directories",
    )

    args = parser.parse_args()

    random.seed(SEED)

    data_input_root_dir = Path(args.data_input_root_dir)
    Path(TRAIN_DATA_DIR).mkdir(parents=True)
    Path(VAL_DATA_DIR).mkdir(parents=True)
    Path(TEST_DATA_DIR).mkdir(parents=True)

    class_directories = [x for x in data_input_root_dir.iterdir()]
    for class_dir in class_directories:
        class_name = class_dir.name
        if class_name == "CT-4":
            continue  # skip CT-4 class because it has only few samples
        nifti_fnames = list(class_dir.glob("*.nii.gz"))

        if class_name == "CT-0":
            class_config = CT_0
        elif class_name == "CT-1":
            class_config = CT_1
        elif class_name == "CT-2":
            class_config = CT_2
        elif class_name == "CT-3":
            class_config = CT_3
        else:
            raise RuntimeError(f"{class_name} is an unknown class name")

        train_fnames = random.sample(nifti_fnames, class_config.TRAIN_SIZE)
        train_fnames = list(
            islice(cycle(train_fnames), class_config.RESAMPLED_TRAIN_SIZE)
        )
        nifti_fnames = [f for f in nifti_fnames if f not in train_fnames]
        val_fnames = random.sample(nifti_fnames, class_config.VAL_SIZE)
        nifti_fnames = [f for f in nifti_fnames if f not in val_fnames]
        test_fnames = random.sample(nifti_fnames, class_config.TEST_SIZE)
        nifti_fnames = [f for f in nifti_fnames if f not in test_fnames]
        print(f"Converting {class_name} train ...")
        convert_to_tfrecord_files(train_fnames, class_config.TRAIN_TFRECORD)
        print(f"Converting {class_name} validation ...")
        convert_to_tfrecord_files(val_fnames, class_config.VAL_TFRECORD)
        print(f"Converting {class_name} test ...")
        convert_to_tfrecord_files(test_fnames, class_config.TEST_TFRECORD)


if __name__ == "__main__":
    main()
