from functools import partial

from tensorflow import keras

SeluConv3D = partial(
    keras.layers.Conv3D,
    padding="same",
    activation="selu",
    kernel_initializer="lecun_normal",
    bias_initializer="zeros",
)

SeluDense = partial(
    keras.layers.Dense,
    activation="selu",
    kernel_initializer="lecun_normal",
    bias_initializer="zeros",
)
