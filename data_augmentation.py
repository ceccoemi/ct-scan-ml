import tensorflow as tf
from scipy import ndimage

from config import SEED


@tf.function
def random_rotate(volume, label):
    """Rotate the volume by a random degree.

    volume must be [z, x, y, (channels)].
    """

    def scipy_rotate(volume):
        angle = tf.random.uniform(
            shape=(1,), minval=-180, maxval=180, dtype=tf.int32, seed=SEED
        )[0].numpy()
        volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume, label


@tf.function
def random_flip(volume, label=None):
    """Flip the volume at a random dimension or leave it unchanged.

    volume must be [z, x, y, (channels)].
    """
    flip_dim = tf.random.uniform(
        (1,), minval=0, maxval=4, dtype=tf.int32, seed=SEED
    )
    if flip_dim[0] != 3:
        volume = tf.reverse(volume, axis=flip_dim)
    return volume, label
