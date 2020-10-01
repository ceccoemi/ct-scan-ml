from tensorflow import keras

from config import input_shape


def conv_block(filters, kernel_size=3, dropout_rate=0.1, pool_size=2):
    """
    - Convolution 3D
    - Selu activation
    - Dropout
    - Max pool 3D
    """
    return (
        keras.layers.Conv3D(
            filters=filters,
            kernel_size=3,
            padding="same",
            kernel_initializer="lecun_normal",
            bias_initializer="lecun_normal",
        ),
        keras.layers.Activation("selu"),
        keras.layers.AlphaDropout(dropout_rate),
        keras.layers.MaxPool3D(pool_size=pool_size),
    )


def deconv_block(filters, kernel_size=3, dropout_rate=0.1, pool_size=2):
    """
    - Up sampling 3D
    - Convolution 3D
    - Selu activation
    - Dropout
    """
    return (
        keras.layers.UpSampling3D(size=pool_size),
        keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="lecun_normal",
            bias_initializer="lecun_normal",
        ),
        keras.layers.Activation("selu"),
        keras.layers.AlphaDropout(dropout_rate),
    )


def build_autoencoder():
    encoder = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape),
            *conv_block(filters=16),
            *conv_block(filters=32),
            *conv_block(filters=64),
        ]
    )
    decoder = keras.models.Sequential(
        [
            keras.layers.InputLayer(
                input_shape=encoder.layers[-1].output.shape[1:],
            ),
            *deconv_block(64),
            *deconv_block(32),
            *deconv_block(16),
            keras.layers.Dense(1),
            keras.layers.Activation("sigmoid"),
        ]
    )
    autoencoder = keras.models.Sequential([encoder, decoder])

    # strategy = tf.distribute.MirroredStrategy()
    # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    #    autoencoder = keras.models.Sequential([encoder, decoder])

    autoencoder.build((None, *input_shape))
    return autoencoder
