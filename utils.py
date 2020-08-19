import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def plot_slice(scan, batch_sample, z):
    """
    Plot a slice of a 3D scan.

    scan must be [batch_size, z, y, x, channels]
    """
    if scan.dtype != tf.float32:
        scan = tf.cast(scan, tf.float32)
    plt.imshow(scan[batch_sample, z, :, :, 0], cmap="gray")


def plot_animated_volume(scan, batch_sample):
    """
    Plot an animation along the z axis.
    
    scan must be [batch_size, z, y, x, channels]
    """
    if scan.dtype != tf.float32:
        scan = tf.cast(scan, tf.float32)

    fig = plt.figure()
    img = plt.imshow(scan[batch_sample, 0, :, :, 0], cmap="gray")
    plt.close()  # to prevent displaying a plot below the video

    def animate(i):
        img.set_array(scan[batch_sample, i, :, :, 0])
        return [img]

    anim = FuncAnimation(
        fig, animate, frames=scan.shape[1], interval=100, blit=True
    )
    return HTML(anim.to_html5_video())
