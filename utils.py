import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def duplicate_iterator(it):
    """Return a generator that duplicate each element of
    the input iterator.

    >>> a = [1, 2, 3]
    >>> list(duplicate_iterator(a))
    [(1, 1), (2, 2), (3, 3)]
    """
    for i in it:
        yield i, i


def plot_slice(scan, z_index=0, ax=None):
    """Plot a slice of a 3D scan.

    scan must be [z, y, x, channels].
    ax is the matplotlib object from which the imshow method will be called.
    """
    if scan.dtype != tf.float32:
        scan = tf.cast(scan, tf.float32)
    if not ax:
        ax = plt
    return ax.imshow(scan[z_index, :, :, 0], cmap="gray")


def plot_animated_volume(scan, fps=10):
    """Plot an animation along the z axis.

    scan must be [z, y, x, channels].
    """
    if scan.dtype != tf.float32:
        scan = tf.cast(scan, tf.float32)

    fig = plt.figure()
    img = plt.imshow(scan[0, :, :, 0], cmap="gray")
    plt.close()  # to prevent displaying a plot below the video

    def animate(i):
        img.set_array(scan[i, :, :, 0])
        return [img]

    anim = FuncAnimation(
        fig,
        animate,
        frames=scan.shape[0],
        interval=1000 / fps,
        blit=True,
    )
    return HTML(anim.to_html5_video())


if __name__ == "__main__":
    import doctest

    doctest.testmod()
