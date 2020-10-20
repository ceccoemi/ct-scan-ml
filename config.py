import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def allocate_gpu_memory_only_when_needed(v):
    "This is to allocate GPU memory only when needed"
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, v)


def compute_precision(precision: str):
    """Set the compute precision.

    Set precision="mixed_float16" to use the mixed precision
    and reduce memory.
    """
    policy = mixed_precision.Policy(precision)
    mixed_precision.set_policy(policy)


use_mixed_precision = False
if use_mixed_precision:
    compute_precision("mixed_float16")

verbose_training = False
seed = 5
data_root_dir = "/pclhcb06/emilio"

# downscale 4
input_shape = (248, 128, 128, 1)
tcia_glob = "tcia-0.25/*.tfrecord"
nrrd_glob = "nrrd-0.25/*.tfrecord"

## downscale 2
# input_shape = (488, 256, 256, 1)
# tcia_glob = "tcia-0.5/*.tfrecord"
# nrrd_glob = "nrrd-0.5/*.tfrecord"

## original
# input_shape = (964, 512, 512, 1)
# tcia_glob = "tcia-0.25/*.tfrecord"
# nrrd_glob = "nrrd-0.25/*.tfrecord"


# Hyperparameters
encoder_num_filters = [32, 64, 128]
epochs = 1000
learning_rate = 0.0001
patience = 20
batch_size = 4
test_num_samples = 10
validation_num_samples = 16  # it should be divisible by batch size
