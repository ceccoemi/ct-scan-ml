from functools import partial

from tensorflow import keras

from config import SMALL_PATCH_SHAPE, BIG_PATCH_SHAPE

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


def build_3d_cnn():
    input_small = keras.Input(SMALL_PATCH_SHAPE, name="input_small")
    x_small = SeluConv3D(
        filters=32,
        kernel_size=3,
        name="small_selu_conv3d_1",
    )(input_small)
    x_small = keras.layers.MaxPooling3D((1, 2, 2), name="small_maxpool_1")(
        x_small
    )
    x_small = SeluConv3D(
        filters=64,
        kernel_size=3,
        name="small_selu_conv3d_2",
    )(x_small)
    x_small = keras.layers.MaxPooling3D((1, 2, 2), name="small_maxpool_2")(
        x_small
    )
    x_small = SeluConv3D(
        filters=128,
        kernel_size=3,
        name="small_selu_conv3d_3",
    )(x_small)
    x_small = keras.layers.MaxPooling3D((1, 2, 2), name="small_maxpool_3")(
        x_small
    )
    x_small = SeluConv3D(
        filters=256,
        kernel_size=3,
        name="small_selu_conv3d_4",
    )(x_small)
    x_small = keras.layers.Flatten(name="flatten_small")(x_small)

    input_big = keras.Input(BIG_PATCH_SHAPE, name="input_big")
    x_big = keras.layers.MaxPooling3D((2, 2, 2), name="big_maxpool_0")(
        input_big
    )
    x_big = SeluConv3D(
        filters=32,
        kernel_size=3,
        name="big_selu_conv3d_1",
    )(x_big)
    x_big = keras.layers.MaxPooling3D((1, 2, 2), name="big_maxpool_1")(x_big)
    x_big = SeluConv3D(
        filters=64,
        kernel_size=3,
        name="big_selu_conv3d_2",
    )(x_big)
    x_big = keras.layers.MaxPooling3D((1, 2, 2), name="big_maxpool_2")(x_big)
    x_big = SeluConv3D(
        filters=128,
        kernel_size=3,
        name="big_selu_conv3d_3",
    )(x_big)
    x_big = keras.layers.MaxPooling3D((1, 2, 2), name="big_maxpool_3")(x_big)
    x_big = SeluConv3D(
        filters=256,
        kernel_size=3,
        name="big_selu_conv3d_4",
    )(x_big)
    x_big = keras.layers.Flatten(name="flatten_big")(x_big)

    x = keras.layers.concatenate([x_small, x_big], name="concatenate")
    x = keras.layers.Dense(1, activation="sigmoid", name="final_dense")(x)

    cnn_3d = keras.Model(
        inputs=[input_small, input_big], outputs=x, name="3dcnn"
    )

    return cnn_3d


def build_pretrained_3d_cnn(model_fname):
    cnn_3d = keras.models.load_model(model_fname)
    cnn_3d.trainable = False
    x = keras.layers.Dense(1, activation="sigmoid", name="final_dense")(
        cnn_3d.layers[-2].output
    )
    return keras.Model(inputs=cnn_3d.input, outputs=x)


def build_pretrained_3d_cnn_with_ae(model_fname):
    small_encoder = keras.models.load_model(model_fname).get_layer("encoder")
    small_encoder._name = "small_encoder"
    for layer in small_encoder.layers:
        layer._name = "small_" + layer._name
    small_encoder.trainable = False

    input_small = keras.Input(SMALL_PATCH_SHAPE, name="input_small")
    x_small = small_encoder(input_small)
    x_small = keras.layers.Flatten(name="flatten_small")(x_small)

    big_encoder = keras.models.load_model(model_fname).get_layer("encoder")
    big_encoder._name = "big_encoder"
    for layer in big_encoder.layers:
        layer._name = "big_" + layer._name
    big_encoder.trainable = False

    input_big = keras.Input(BIG_PATCH_SHAPE, name="input_big")
    x_big = keras.layers.MaxPooling3D((2, 2, 2), name="big_maxpool_0")(
        input_big
    )
    x_big = big_encoder(x_big)
    x_big = keras.layers.Flatten(name="flatten_big")(x_big)

    x = keras.layers.concatenate([x_small, x_big], name="concatenate")

    x = keras.layers.Dense(1, activation="sigmoid", name="final_dense")(x)

    cnn_3d = keras.Model(
        inputs=[input_small, input_big], outputs=x, name="pretrained-3dcnn"
    )

    return cnn_3d
