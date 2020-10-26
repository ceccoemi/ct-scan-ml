from tensorflow import keras


def conv_block(x, filters, kernel_size=3, dropout_rate=0.1, pool_size=2):
    """
    - Convolution 3D (with selu activation)
    - AlphaDropout
    - Max pool 3D

    x is the input layer using the Keras Functional API
    """
    x = keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="lecun_normal",
        bias_initializer="lecun_normal",
        activation="selu",
    )(x)
    x = keras.layers.AlphaDropout(dropout_rate)(x)
    x = keras.layers.MaxPool3D(pool_size=pool_size)(x)
    return x


def deconv_block(x, filters, kernel_size=3, dropout_rate=0.1, pool_size=2):
    """
    - Up sampling 3D
    - Convolution 3D (with selu)
    - AlphaDropout

    x is the input layer using the Keras Functional  API
    """
    x = keras.layers.UpSampling3D(size=pool_size)(x)
    x = keras.layers.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="lecun_normal",
        bias_initializer="lecun_normal",
        activation="selu",
    )(x)
    x = keras.layers.AlphaDropout(dropout_rate)(x)
    return x
