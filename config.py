verbose_training = False

# downscale 4
input_shape = (248, 128, 128, 1)
tcia_glob = "tcia-0.25/*.tfrecord"
nrrd_glob = "nrrd-0.25/*.tfrecord"

## downscale 2
# input_shape = (488, 256, 256, 1)
# tcia_glob = "icia-0.5/*.tfrecords"
# nrrd_glob = "nrrd-0.5/*.tfrecords"

## original
# input_shape = (964, 512, 512, 1)
# tcia_glob = "tcia-0.25/*.tfrecords"
# nrrd_glob = "nrrd-0.25/*.tfrecords"

# Hyperparameters
epochs = 1000
learning_rate = 0.0001
patience = 20
batch_size = 2
test_num_samples = 10
validation_num_samples = 10  # it should be divisible by batch size
