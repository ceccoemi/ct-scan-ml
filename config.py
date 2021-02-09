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

COVID_NEG_TFRECORD = f"{DATA_ROOT_DIR}/covid_neg.tfrecord"
COVID_POS_TFRECORD = f"{DATA_ROOT_DIR}/covid_pos.tfrecord"

COVID_SCAN_SHAPE = (42, 256, 256, 1)

########### Num nodules regression #########

LIDC_TOT_NUM_NODULES_TFRECORD = (
    f"{DATA_ROOT_DIR}/lidc_tot_num_nodules.tfrecord"
)
LIDC_NUM_BIG_NODULES_TFRECORD = (
    f"{DATA_ROOT_DIR}/lidc_num_big_nodules.tfrecord"
)

RESIZED_SCAN_SHAPE = (128, 256, 256, 1)
SCAN_SHAPE = (112, 192, 224, 1)
