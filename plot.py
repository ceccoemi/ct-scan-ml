import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


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


def grid_plot_slices(scan):
    """Plot multiple slices of a 3D scan.

    scan must be [z, y, x, channels].
    """
    fig, ax = plt.subplots(nrows=3, ncols=4)
    z = 0
    for row in range(1, 4):
        for col in range (1, 5):
            plot_slice(scan[z, :], ax[row][col])
            z += 1


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
