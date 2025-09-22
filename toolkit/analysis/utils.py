import numpy as np


def array_spacing(x):
    """Get spacing of an evenly spaced array."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if x.size < 2:
        raise ValueError("Input must have at least 2 elements.")
    return (x[-1] - x[0]) / x.size


