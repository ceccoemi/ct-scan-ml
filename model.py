from tensorflow import keras

from config import input_shape, encoder_num_filters


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


def build_encoder(
    encoder_input_shape=input_shape, num_filters=encoder_num_filters
):
    """Return the encoder model.

    encoder_num_filters is a list of the number of filters
    of each conv_block of the encoder.
    """
    encoder_inputs = keras.Input(encoder_input_shape)
    x = encoder_inputs
    for f in num_filters:
        x = conv_block(x, filters=f)
    encoder_outputs = x
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")
    return encoder


def build_autoencoder(
    encoder_input_shape=input_shape, encoder_num_filters=encoder_num_filters
):
    """Build the autoencoder (encoder + decoder).

    encoder_num_filters is a list of the number of filters
    of each conv_block of the encoder.
    The decoder is a mirrored image of the encoder
    plus a dense layer at the end with one neuron.
    """
    encoder = build_encoder(encoder_input_shape, encoder_num_filters)
    decoder_inputs = keras.Input(encoder.output_shape[1:])
    x = decoder_inputs
    for num_filters in reversed(encoder_num_filters):
        x = deconv_block(x, filters=num_filters)
    x = keras.layers.Dense(1)(x)
    decoder_outputs = keras.layers.Activation("sigmoid", dtype="float32")(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
    autoencoder = keras.models.Sequential(
        [encoder, decoder], name="autoencoder"
    )
    assert autoencoder.output_shape[1:] == encoder_input_shape, (
        "Autoencoder input and output must have the same shape, "
        f"expected {encoder_input_shape}; actual: {autoencoder.output_shape[1:]}"
    )
    return autoencoder
