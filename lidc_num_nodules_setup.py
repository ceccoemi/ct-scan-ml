import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import ndimage
from tqdm import tqdm

from utils import read_dcm, volume_to_example
from config import (
    LIDC_TOT_NUM_NODULES_TFRECORD,
    LIDC_NUM_BIG_NODULES_TFRECORD,
    RESIZED_SCAN_SHAPE,
    SCAN_SHAPE,
)


def read_xlsx_nodule_counts(xlsx_file):
    "Read and transform the Excel file containing the nodule counts"
    df = (
        pd.read_excel("/pclhcb06/emilio/lidc-nodule-counts.xlsx")
        .drop(columns=["Unnamed: 4", "Unnamed: 5"])
        .dropna()
        .rename(
            columns={
                "TCIA Patent ID": "patient_id",
                "Total Number of Nodules* ": "total_nodules",
                "Number of Nodules >=3mm**": "big_nodules",
                "Number of Nodules <3mm***": "small_nodules",
            }
        )
    )
    df["patient_id"] = df["patient_id"].apply(lambda x: x[10:]) # Remove "LIDC-IDRI-" prefix
    df = df.drop_duplicates("patient_id").reset_index()
    return df


def preprocess_scan(scan):
    z, y, x = scan.shape
    target_z, target_y, target_x, _ = RESIZED_SCAN_SHAPE
    scan = ndimage.zoom(
        scan, (target_z / z, target_y / y, target_x / x), order=5
    )
    scan = scan.astype(np.float32)
    scan = scan[8:-8, 32:-32, 16:-16]
    scan = np.expand_dims(scan, axis=-1)  # add the channel dimension
    assert scan.shape == SCAN_SHAPE
    return scan


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create two TFRecords with entries of the type (ct_scan, num_nodules): "
            "one with the total number of nodules and one with the number of nodules with size >=3mm"
        )
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing all the DCM files downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
    )
    parser.add_argument(
        "xlsx_file",
        help="Excel file downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI, which contains the counts of the nodules",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = read_xlsx_nodule_counts(args.xlsx_file)
    with \
        tf.io.TFRecordWriter(LIDC_TOT_NUM_NODULES_TFRECORD) as tot_writer, \
        tf.io.TFRecordWriter(LIDC_NUM_BIG_NODULES_TFRECORD) as big_writer \
    :
        for row in tqdm(df.itertuples(), total=len(df.index)):
            dcm_dirs = list(data_dir.glob(f"LIDC-IDRI-{row.patient_id}/*/*/"))
            # Take the directory with the largest number of elements,
            # which is the directory with the greates number of DICOM slices
            dcm_dir = sorted(
                dcm_dirs, key=lambda x: len(list(x.glob("*"))), reverse=True
            )[0]
            scan = read_dcm(dcm_dir)
            scan = preprocess_scan(scan)
            tot_scan_example = volume_to_example(scan, label=row.total_nodules)
            tot_writer.write(tot_scan_example.SerializeToString())
            big_scan_example = volume_to_example(scan, label=row.big_nodules)
            big_writer.write(big_scan_example.SerializeToString())


if __name__ == "__main__":
    main()
