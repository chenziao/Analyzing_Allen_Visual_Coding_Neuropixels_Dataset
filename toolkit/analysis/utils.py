import numpy as np

from numpy.typing import ArrayLike, NDArray


def array_spacing(x : ArrayLike) -> float:
    """Get spacing of an evenly spaced array."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if x.size < 2:
        raise ValueError("Input must have at least 2 elements.")
    return (x[-1] - x[0]) / (x.size - 1)


def get_bins(window : tuple[float, float] = (0., 1.), bin_width : float = 1.0) -> NDArray[float]:
    """Get bins for a given bin width within a window aligning one bin center at zero.
    Note that the actual window edges are rounded to the nearest bin width.
    
    Parameters
    ----------
    window : tuple[float, float]
        Window.
    bin_width : float
        Bin width.

    Returns
    -------
    bin_centers : NDArray[float]
        Bin centers.
    bin_edges : NDArray[float]
        Bin edges.
    """
    int_window = np.round(np.array(window) / bin_width).astype(int)
    bin_centers = np.arange(int_window[0], int_window[1] + 1) * bin_width
    bin_edges = np.append(bin_centers - bin_width / 2, bin_centers[-1] + bin_width / 2)
    return bin_centers, bin_edges

