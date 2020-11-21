DATA_ROOT_DIR = "/pclhcb06/emilio"

LIDC_SMALL_NEG_TFRECORD = f"{DATA_ROOT_DIR}/lidc_small_neg.tfrecord"
LIDC_SMALL_POS_TFRECORD = f"{DATA_ROOT_DIR}/lidc_small_pos.tfrecord"
LIDC_BIG_NEG_TFRECORD = f"{DATA_ROOT_DIR}/lidc_big_neg.tfrecord"
LIDC_BIG_POS_TFRECORD = f"{DATA_ROOT_DIR}/lidc_big_pos.tfrecord"
LIDC_SMALL_UNLABELED_TFRECORD = (
    f"{DATA_ROOT_DIR}/lidc_small_unlabeled.tfrecord"
)
LIDC_BIG_UNLABELED_TFRECORD = f"{DATA_ROOT_DIR}/lidc_big_unlabeled.tfrecord"

SPIE_SMALL_NEG_TFRECORD = f"{DATA_ROOT_DIR}/spie_small_neg.tfrecord"
SPIE_SMALL_POS_TFRECORD = f"{DATA_ROOT_DIR}/spie_small_pos.tfrecord"
SPIE_BIG_NEG_TFRECORD = f"{DATA_ROOT_DIR}/spie_big_neg.tfrecord"
SPIE_BIG_POS_TFRECORD = f"{DATA_ROOT_DIR}/spie_big_pos.tfrecord"

SEED = 5
SMALL_PATCH_SHAPE = (8, 32, 32, 1)
BIG_PATCH_SHAPE = (16, 64, 64, 1)
