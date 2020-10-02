from tensorflow import keras

from config import input_shape


def conv_block(x, filters, kernel_size=3, dropout_rate=0.1, pool_size=2):
    """
    - Convolution 3D (with selu)
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


def build_autoencoder():
    encoder_inputs = keras.Input(input_shape)
    x = conv_block(encoder_inputs, filters=16)
    x = conv_block(x, filters=32)
    encoder_outputs = conv_block(x, filters=64)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")

    decoder_inputs = keras.Input(encoder.output_shape[1:])
    x = deconv_block(decoder_inputs, filters=64)
    x = deconv_block(x, filters=32)
    x = deconv_block(x, filters=16)
    decoder_outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    autoencoder = keras.models.Sequential([encoder, decoder])

    assert (
        autoencoder.output_shape[1:] == input_shape
    ), "Autoencoder input and output must have the same shape"

    return autoencoder
