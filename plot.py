import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def _validate_volume(volume):
    assert (
        3 <= len(volume.shape) <= 4
    ), f"volume must be [z, y, x, (channels)], where channels is optional. Get {volume.shape}"
    if tf.is_tensor(volume):
        volume = volume.numpy()
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)
    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]  # take the first channel
    return volume


def plot_slice(volume, axis="z", index=0, ax=None):
    """Plot a slice of a 3D volume along the specified axis.

    volume must be [z, y, x, (channels)], where channels is optional.
    In case of multiple channels, it takes the first one.
    axis can be "z", "y" or "x".
    ax is the matplotlib object from which the imshow method will be called.
    """
    volume = _validate_volume(volume)
    if not ax:
        ax = plt
    if axis == "z":
        volume_to_plot = volume[index, :, :]
    elif axis == "y":
        volume_to_plot = volume[:, index, :]
    elif axis == "x":
        volume_to_plot = volume[:, :, index]
    else:
        raise ValueError("axis can be 'z', 'y' or 'x'.")
    return ax.imshow(volume_to_plot, cmap="gray")


def plot_volume_animation(volume, axis="z", fps=10):
    """Plot an animation of the volume along the specified axis.

    volume must be [z, y, x, (channels)], where channels is optional.
    In case of multiple channels, it takes the first one.
    axis can be "z", "y" or "x".
    Return a video that can be displayed in a Jupyter Notebook.
    """
    volume = _validate_volume(volume)
    if axis == "z":
        return _z_animation(volume, fps)
    elif axis == "y":
        return _y_animation(volume, fps)
    elif axis == "x":
        return _x_animation(volume, fps)
    else:
        raise ValueError("axis can be 'z', 'y' or 'x'.")


def _z_animation(volume, fps):
    fig = plt.figure()
    img = plt.imshow(volume[0, :, :], cmap="gray")
    plt.close()  # to prevent displaying a plot below the video

    def animate(i):
        img.set_array(volume[i, :, :])
        return [img]

    anim = FuncAnimation(
        fig,
        animate,
        frames=volume.shape[0],
        interval=1000 / fps,
        blit=True,
    )
    return HTML(anim.to_html5_video())


def _y_animation(volume, fps):
    fig = plt.figure()
    img = plt.imshow(volume[:, 0, :], cmap="gray")
    plt.close()  # to prevent displaying a plot below the video

    def animate(i):
        img.set_array(volume[:, i, :])
        return [img]

    anim = FuncAnimation(
        fig,
        animate,
        frames=volume.shape[1],
        interval=1000 / fps,
        blit=True,
    )
    return HTML(anim.to_html5_video())


def _x_animation(volume, fps):
    fig = plt.figure()
    img = plt.imshow(volume[:, :, 0], cmap="gray")
    plt.close()  # to prevent displaying a plot below the video

    def animate(i):
        img.set_array(volume[:, :, i])
        return [img]

    anim = FuncAnimation(
        fig,
        animate,
        frames=volume.shape[2],
        interval=1000 / fps,
        blit=True,
    )
    return HTML(anim.to_html5_video())
