import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils import read_dcm, extract_patch, pad_to_shape, volume_to_example
from config import (
    SPIE_SMALL_NEG_TFRECORD,
    SPIE_BIG_NEG_TFRECORD,
    SPIE_SMALL_POS_TFRECORD,
    SPIE_BIG_POS_TFRECORD,
    SMALL_PATCH_SHAPE,
    BIG_PATCH_SHAPE,
)


def read_xls(xls_file):
    "Read the excel file with nodule locations and labels"
    df = pd.read_excel(xls_file)
    df = df[:73]
    df[["xloc", "yloc"]] = df["Nodule Center x,y Position*"].apply(
        lambda x: pd.Series(eval(x))
    )
    df["Final Diagnosis"] = df["Final Diagnosis"].apply(
        lambda x: "0" if x == "Benign nodule" else "1"
    )
    df = df.drop(columns=["Nodule Number", "Nodule Center x,y Position*"])
    df = df.rename(
        columns={
            "Scan Number": "scan_number",
            "Nodule Center Image": "zloc",
            "Final Diagnosis": "label",
        }
    )
    df["zloc"] = df["zloc"].astype(int)
    df["label"] = df["label"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract the 3D patches containing the nodules and store them in TFRecord files.",
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing all the DCM files downloaded from https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge",
    )
    parser.add_argument(
        "xls_file",
        help="Excel file obtained from https://wiki.cancerimagingarchive.net/display/Public/SPIE-AAPM+Lung+CT+Challenge",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = read_xls(args.xls_file)
    assert (
        len(df.index) == 73
    ), f"The input excel {args.xls_file} has not the expected size."

    with tf.io.TFRecordWriter(
        SPIE_SMALL_NEG_TFRECORD
    ) as small_neg_writer, tf.io.TFRecordWriter(
        SPIE_SMALL_POS_TFRECORD
    ) as small_pos_writer, tf.io.TFRecordWriter(
        SPIE_BIG_NEG_TFRECORD
    ) as big_neg_writer, tf.io.TFRecordWriter(
        SPIE_BIG_POS_TFRECORD
    ) as big_pos_writer:
        for row in tqdm(df.itertuples(), total=len(df.index)):
            dcm_dir_glob = Path(data_dir).glob(f"{row.scan_number}/*/*/")
            dcm_dir = list(dcm_dir_glob)[0]
            scan = read_dcm(dcm_dir, reverse_z=False)

            big_writer = big_pos_writer if row.label else big_neg_writer
            big_patch = extract_patch(
                scan,
                (row.zloc, row.yloc, row.xloc),
                BIG_PATCH_SHAPE[:-1],
            )
            big_patch = pad_to_shape(big_patch, BIG_PATCH_SHAPE[:-1])
            big_patch = np.expand_dims(big_patch, axis=-1)
            if not big_patch.any():
                print(
                    f"WARNING ({row.scan_number=}): "
                    "Patch contains only zeros. Skipping this patch ..."
                )
                continue
            assert (
                big_patch.shape == BIG_PATCH_SHAPE
            ), f"Wrong shape for scan {row.scan_number}."
            big_patch = big_patch.astype(np.float32)
            big_example = volume_to_example(big_patch)
            big_writer.write(big_example.SerializeToString())

            small_writer = small_pos_writer if row.label else small_neg_writer
            small_patch = extract_patch(
                scan,
                (row.zloc, row.yloc, row.xloc),
                SMALL_PATCH_SHAPE[:-1],
            )
            small_patch = pad_to_shape(small_patch, SMALL_PATCH_SHAPE[:-1])
            small_patch = np.expand_dims(small_patch, axis=-1)
            assert (
                small_patch.shape == SMALL_PATCH_SHAPE
            ), f"Wrong shape for scan {row.scan_number}."
            small_patch = small_patch.astype(np.float32)
            small_example = volume_to_example(small_patch)
            small_writer.write(small_example.SerializeToString())


if __name__ == "__main__":
    main()
