import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from statistics import median

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils import read_dcm, extract_patch, pad_to_shape, volume_to_example
from config import (
    LIDC_SMALL_NEG_TFRECORD,
    LIDC_SMALL_POS_TFRECORD,
    LIDC_BIG_NEG_TFRECORD,
    LIDC_BIG_POS_TFRECORD,
    LIDC_SMALL_UNLABELED_TFRECORD,
    LIDC_BIG_UNLABELED_TFRECORD,
    SMALL_PATCH_SHAPE,
    BIG_PATCH_SHAPE,
)


def read_lidc_size_report(csv_file):
    "Read the CSV file obtained from  http://www.via.cornell.edu/lidc/"
    df = pd.read_csv(csv_file, dtype={"case": str, "scan": str})
    df["noduleIDs"] = (
        df[["nodIDs", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12"]]
        .fillna("")
        .values.tolist()
    )
    df["noduleIDs"] = df["noduleIDs"].apply(lambda x: [e for e in x if e])
    df = df.drop(
        columns=[
            "volume",
            "eq. diam.",
            "nodIDs",
            "Unnamed: 8",
            "Unnamed: 10",
            "Unnamed: 11",
            "Unnamed: 12",
            "Unnamed: 13",
            "Unnamed: 14",
            "Unnamed: 15",
        ]
    ).rename(
        columns={
            "x loc.": "xloc",
            "y loc.": "yloc",
            "slice no.": "zloc",
            "noduleIDs": "ids",
        }
    )
    df = df[
        df.ids.apply(len) >= 3
    ]  # keep the scan with at least 3 evaluations
    return df


def get_malignancies(xml_file, nodule_ids):
    "Return a list of the assigned malignancies extracted from the XML"
    tree = ET.parse(xml_file)
    root = tree.getroot()
    prefix = "{http://www.nih.gov}"
    malignancies = []
    for reading_session in root.findall(f"{prefix}readingSession"):
        for nodule in reading_session.findall(f"{prefix}unblindedReadNodule"):
            nodule_id = nodule.findall(f"{prefix}noduleID")[0].text
            if nodule_id in nodule_ids:
                malignancy = int(
                    nodule.findall(f"*/{prefix}malignancy")[0].text
                )
                if malignancy > 0:
                    malignancies.append(malignancy)
    return malignancies


def main():
    parser = argparse.ArgumentParser(
        description="Extract the 3D patches containing the nodules and store them in TFRecord files.",
    )
    parser.add_argument(
        "data_dir",
        help="Directory containing all the DCM files downloaded from https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
    )
    parser.add_argument(
        "csv_file",
        help="CSV file obtained from http://www.via.cornell.edu/lidc",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    nodules_df = read_lidc_size_report(args.csv_file)
    assert (
        len(nodules_df.index) == 1387
    ), f"The input CSV {args.csv_file} has not the expected size."

    with tf.io.TFRecordWriter(
        LIDC_SMALL_NEG_TFRECORD
    ) as small_neg_writer, tf.io.TFRecordWriter(
        LIDC_SMALL_POS_TFRECORD
    ) as small_pos_writer, tf.io.TFRecordWriter(
        LIDC_BIG_NEG_TFRECORD
    ) as big_neg_writer, tf.io.TFRecordWriter(
        LIDC_BIG_POS_TFRECORD
    ) as big_pos_writer, tf.io.TFRecordWriter(
        LIDC_SMALL_UNLABELED_TFRECORD
    ) as small_unlabeled_writer, tf.io.TFRecordWriter(
        LIDC_BIG_UNLABELED_TFRECORD
    ) as big_unlabeled_writer:
        for row in tqdm(nodules_df.itertuples(), total=len(nodules_df.index)):
            case = row.case
            scan_id = row.scan
            dcm_dir_glob = list(
                data_dir.glob(f"LIDC-IDRI-{case}/*/{scan_id}.*/")
            )
            if len(dcm_dir_glob) == 0:
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "Scan not found. Skipping this scan ..."
                )
                continue
            if len(dcm_dir_glob) > 1:
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "Found multiple scans with same ids. Skipping this scan ..."
                )
                continue
            dcm_dir = dcm_dir_glob[0]
            scan = read_dcm(dcm_dir, reverse_z=True)
            xml_files = list(dcm_dir.glob("*.xml"))
            if len(xml_files) == 0:
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "Can't find a XML file. Skipping this scan ..."
                )
                continue
            elif len(xml_files) > 1:
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "Found multiple XML files. Skipping this scan ..."
                )
                continue
            xml_file = xml_files[0]
            nodule_ids = row.ids
            malignancies = get_malignancies(xml_file, nodule_ids)
            median_malignancy = median(malignancies)
            if median_malignancy < 3:
                big_writer = big_neg_writer
                small_writer = small_neg_writer
            elif median_malignancy > 3:
                big_writer = big_pos_writer
                small_writer = small_pos_writer
            else:
                # if the malignancies median is 3 then write the patch
                # as unlabeled
                big_writer = big_unlabeled_writer
                small_writer = small_unlabeled_writer

            big_patch = extract_patch(
                scan,
                (row.zloc, row.yloc, row.xloc),
                BIG_PATCH_SHAPE[:-1],
            )
            big_patch = pad_to_shape(big_patch, BIG_PATCH_SHAPE[:-1])
            big_patch = np.expand_dims(big_patch, axis=-1)
            if not big_patch.any():
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "Patch contains only zeros. Skipping this patch ..."
                )
                continue
            assert (
                big_patch.shape == BIG_PATCH_SHAPE
            ), f"Wrong shape for scan {scan_id} in case {case}."
            big_patch = big_patch.astype(np.float32)
            big_example = volume_to_example(big_patch)
            big_writer.write(big_example.SerializeToString())

            small_patch = extract_patch(
                scan,
                (row.zloc, row.yloc, row.xloc),
                SMALL_PATCH_SHAPE[:-1],
            )
            small_patch = pad_to_shape(small_patch, SMALL_PATCH_SHAPE[:-1])
            small_patch = np.expand_dims(small_patch, axis=-1)
            assert (
                small_patch.shape == SMALL_PATCH_SHAPE
            ), f"Wrong shape for scan {scan_id} in case {case}."
            small_patch = small_patch.astype(np.float32)
            small_example = volume_to_example(small_patch)
            small_writer.write(small_example.SerializeToString())


if __name__ == "__main__":
    main()
