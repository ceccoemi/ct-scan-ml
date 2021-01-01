DATA_ROOT_DIR = "/pclhcb06/emilio"

########## Lung nodule classification ##########

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

SMALL_PATCH_SHAPE = (8, 32, 32, 1)
BIG_PATCH_SHAPE = (16, 64, 64, 1)

########## Covid classification ##########

CT_0_TFRECORD = f"{DATA_ROOT_DIR}/CT-0.tfrecord"
CT_1_TFRECORD = f"{DATA_ROOT_DIR}/CT-1.tfrecord"
CT_2_TFRECORD = f"{DATA_ROOT_DIR}/CT-2.tfrecord"
CT_3_TFRECORD = f"{DATA_ROOT_DIR}/CT-3.tfrecord"
CT_4_TFRECORD = f"{DATA_ROOT_DIR}/CT-4.tfrecord"
# CT_0_TFRECORD = f"{DATA_ROOT_DIR}/CT-0-0.25.tfrecord"
# CT_1_TFRECORD = f"{DATA_ROOT_DIR}/CT-1-0.25.tfrecord"
# CT_2_TFRECORD = f"{DATA_ROOT_DIR}/CT-2-0.25.tfrecord"
# CT_3_TFRECORD = f"{DATA_ROOT_DIR}/CT-3-0.25.tfrecord"
# CT_4_TFRECORD = f"{DATA_ROOT_DIR}/CT-4-0.25.tfrecord"

COVID_NEG_TFRECORD = f"{DATA_ROOT_DIR}/covid_neg.tfrecord"
COVID_POS_TFRECORD = f"{DATA_ROOT_DIR}/covid_pos.tfrecord"
# COVID_NEG_TFRECORD = f"{DATA_ROOT_DIR}/covid_neg_0.25.tfrecord"
# COVID_POS_TFRECORD = f"{DATA_ROOT_DIR}/covid_pos_0.25.tfrecord"

RESIZED_SCAN_SHAPE = (64, 256, 256, 1)
# RESIZED_SCAN_SHAPE = (64, 128, 128, 1)
SCAN_SHAPE = (64, 192, 224, 1)
# SCAN_SHAPE = (64, 96, 112, 1)

########### Num nodules regression #########

LIDC_NUM_NODULES_TFRECORD = f"{DATA_ROOT_DIR}/lidc_num_nodules.tfrecord"
