import numpy as np

from numpy.typing import ArrayLike, NDArray


def spike_count(tspk : ArrayLike, bin_edges : NDArray[float]) -> NDArray[int]:
    """Count spikes given in an array of spike times into bins.
    
    Parameters
    ----------
    tspk : ArrayLike
        Array of spike times.
    bin_edges : NDArray[float]
        Time bins edges.

    Returns
    -------
    cspk : NDArray[int]
        Array of spike counts in each bin.
    """
    ispk = np.digitize(tspk, bin_edges)
    cspk = np.bincount(ispk, minlength=bin_edges.size + 1)
    return cspk[1:bin_edges.size]


