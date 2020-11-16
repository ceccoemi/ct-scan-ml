import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from math import ceil, floor
from statistics import median

import numpy as np
import pandas as pd
from pydicom import dcmread
import tensorflow as tf
from tqdm import tqdm

from config import (
    SMALL_NEG_TFRECORD,
    SMALL_POS_TFRECORD,
    BIG_NEG_TFRECORD,
    BIG_POS_TFRECORD,
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
        SMALL_NEG_TFRECORD
    ) as small_neg_writer, tf.io.TFRecordWriter(
        SMALL_POS_TFRECORD
    ) as small_pos_writer, tf.io.TFRecordWriter(
        BIG_NEG_TFRECORD
    ) as big_neg_writer, tf.io.TFRecordWriter(
        BIG_POS_TFRECORD
    ) as big_pos_writer:
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
            scan = read_dcm(dcm_dir)
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
            if len(malignancies) < 3:
                print(
                    f"WARNING ({scan_id=} {case=}): "
                    "There are less than 3 evaluations. Skipping this patch ..."
                )
            median_malignancy = median(malignancies)
            if median_malignancy < 3:
                malignancy = 0
            elif median_malignancy > 3:
                malignancy = 1
            else:
                continue  # if the malignancies median is 3 then discard the nodule

            big_writer = big_pos_writer if malignancy else big_neg_writer
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

            small_writer = small_pos_writer if malignancy else small_neg_writer
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
