DATA_ROOT_DIR = "/pclhcb06/emilio"
TRAIN_DATA_DIR = f"{DATA_ROOT_DIR}/MosMed-train"
VAL_DATA_DIR = f"{DATA_ROOT_DIR}/MosMed-val"
TEST_DATA_DIR = f"{DATA_ROOT_DIR}/MosMed-test"

SEED = 5

SCAN_SHAPE = [48, 256, 256]


class CT_0:
    NAME = "CT-0"

    TRAIN_SIZE = 164
    RESAMPLED_TRAIN_SIZE = 230  # +40%
    VAL_SIZE = 40
    TEST_SIZE = 50

    TRAIN_TFRECORD = f"{TRAIN_DATA_DIR}/CT-0.tfrecord"
    VAL_TFRECORD = f"{VAL_DATA_DIR}/CT-0.tfrecord"
    TEST_TFRECORD = f"{TEST_DATA_DIR}/CT-0.tfrecord"


class CT_1:
    NAME = "CT-1"

    TRAIN_SIZE = 438
    RESAMPLED_TRAIN_SIZE = 307  # -30%
    VAL_SIZE = 110
    TEST_SIZE = 136

    TRAIN_TFRECORD = f"{TRAIN_DATA_DIR}/CT-1.tfrecord"
    VAL_TFRECORD = f"{VAL_DATA_DIR}/CT-1.tfrecord"
    TEST_TFRECORD = f"{TEST_DATA_DIR}/CT-1.tfrecord"


class CT_2:
    NAME = "CT-2"

    TRAIN_SIZE = 80
    RESAMPLED_TRAIN_SIZE = 198  # +148%
    VAL_SIZE = 20
    TEST_SIZE = 25

    TRAIN_TFRECORD = f"{TRAIN_DATA_DIR}/CT-2.tfrecord"
    VAL_TFRECORD = f"{VAL_DATA_DIR}/CT-2.tfrecord"
    TEST_TFRECORD = f"{TEST_DATA_DIR}/CT-2.tfrecord"


class CT_3:
    NAME = "CT-3"

    TRAIN_SIZE = 29
    RESAMPLED_TRAIN_SIZE = 168  # +479%
    VAL_SIZE = 7
    TEST_SIZE = 9

    TRAIN_TFRECORD = f"{TRAIN_DATA_DIR}/CT-3.tfrecord"
    VAL_TFRECORD = f"{VAL_DATA_DIR}/CT-3.tfrecord"
    TEST_TFRECORD = f"{TEST_DATA_DIR}/CT-3.tfrecord"
