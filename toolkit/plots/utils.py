import numpy as np
import matplotlib.pyplot as plt

from numpy.typing import ArrayLike, NDArray


def set_equal_3d_scaling(ax : plt.Axes) -> NDArray[float]:
    """Set equal scaling for 3D plot in a cubic volume

    Returns
    -------
    limits : NDArray[float]
        Limits of the axes. Shape: (3, 2).
    """
    # Get current ranges
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    # Calculate centers
    center = np.mean(limits, axis=1, keepdims=True)
    # Calculate cubic range
    half_range = np.max(limits[:, 1] - limits[:, 0]) / 2
    limits = center + half_range * np.array([-1, 1])
    # Set limits
    ax.set_xlim3d(limits[0])
    ax.set_ylim3d(limits[1])
    ax.set_zlim3d(limits[2])
    ax.set_box_aspect([1, 1, 1])
    return limits


def lighten(val, clr, light_scale=0.7, dark_scale=0.6):
    """Change color lightness by value between [0, 1]"""
    clr1 = 1 - (1 - clr) * (1 - light_scale)
    clr2 = clr * (1 - dark_scale)
    val = np.asarray(val)
    if val.size > 1:
        val = val[:, None]
        clr1, clr2 = clr1[None, :], clr2[None, :]
    return (1 - val) * clr1 + val * clr2


def get_lighten_cmap(clr, N=16, light_scale=0.2, dark_scale=0.8, revert=False):
    """Get colormap with varying lightness of a given color"""
    from matplotlib.colors import LinearSegmentedColormap
    value = np.linspace(1, 0, N) if revert else np.linspace(0, 1, N)
    lighten_cmap = LinearSegmentedColormap.from_list(
        'lighten', lighten(value, clr, light_scale, dark_scale))
    return lighten_cmap


def plot_multicolor_line(*args, c=None, ax=None, cmap='jet', linewidth=2, linestyle='-', alpha=1):
    """Plot line with varying color on its segments. Input coordinates: x, y, (z, optional)
    args: arrays of coordinates of points
    c: values for color map, corresponding to points or segments
    """
    points = np.column_stack(args)[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if c is None:
        c = np.arange(points.shape[0])
    cmid = (c[:-1] + c[1:]) / 2 if len(c) == points.shape[0] else c
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(c.min(), c.max())
    if segments.shape[-1] > 2:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        linecollection = Line3DCollection
    else:
        from matplotlib.collections import LineCollection
        linecollection = LineCollection
    lc = linecollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(cmid)
    lc.set_linewidth(linewidth)
    lc.set_linestyle(linestyle)
    lc.set_alpha(alpha)
    if ax is None:
        _, ax = plt.subplots()
    line = ax.add_collection(lc)
    return line

