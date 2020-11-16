DATA_ROOT_DIR = "/pclhcb06/emilio"
SMALL_NEG_TFRECORD = f"{DATA_ROOT_DIR}/small_neg_nodules.tfrecord"
SMALL_POS_TFRECORD = f"{DATA_ROOT_DIR}/small_pos_nodules.tfrecord"
BIG_NEG_TFRECORD = f"{DATA_ROOT_DIR}/big_neg_nodules.tfrecord"
BIG_POS_TFRECORD = f"{DATA_ROOT_DIR}/big_pos_nodules.tfrecord"

SEED = 5
SMALL_PATCH_SHAPE = (8, 56, 56, 1)
BIG_PATCH_SHAPE = (16, 112, 112, 1)
