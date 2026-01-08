import numpy as np
import xarray as xr

from .signal import gaussian_filter1d_da

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


def smooth_spike_rate(
    spike_rate : xr.DataArray,
    sigma : float,
    mean_spike_rate : ArrayLike | float | None = None,
    normalization_scale : ArrayLike | float | str | None = None,
    soft_normalize_cut : float = 0.0,
    time_dim : str = 'time',
    unit_dim : str = 'unit_id'
) -> xr.DataArray:
    """Smooth spike rate with Gaussian filter and optionally normalize
    
    Parameters
    ----------
    spike_rate : xr.DataArray
        Spike rate data. Should 
    sigma : float
        Sigma of the Gaussian filter.
    mean_spike_rate : ArrayLike | float | None
        Mean spike rate. If not specified, compute the mean from `spike_rate`
        along all dimensions other than `unit_dim`.
    normalization_scale : ArrayLike | float | str | None
        Normalization scale to divide the spike rate. If not specified, no normalization is performed.
        If is a string, it is treated as a statistic method name (e.g. 'std') and
        the normalization scale is computed from `spike_rate` along all dimensions other than `unit_dim`.
    soft_normalize_cut : float
        Soft normalization cutoff. Normalized rate = spike_rate / (normalization_scale + soft_normalize_cut)
    time_dim : str
        Time dimension to filter along.
    unit_dim : str
        Unit dimension used to determine other dimensions to get statistics.

    Returns
    -------
    smoothed : xr.DataArray
        Smoothed spike rate.
    """
    dims = [d for d in spike_rate.dims if d != unit_dim]  # dimensions other than unit_dim
    if unit_dim in spike_rate.coords:
        unit_coords = {unit_dim: spike_rate.coords[unit_dim]}
    else:
        unit_coords = {unit_dim: np.array(0)}
    unit_shape = unit_coords[unit_dim].shape
    if mean_spike_rate is None:
        mean_spike_rate = spike_rate.mean(dim=dims)
    else:  # scalar or array
        mean_spike_rate = np.broadcast_to(mean_spike_rate, unit_shape)
        mean_spike_rate = xr.DataArray(mean_spike_rate, coords=unit_coords)

    # smooth firing rate using Gaussian filter (constant mode filling mean spike rate value)
    smoothed = gaussian_filter1d_da(spike_rate - mean_spike_rate, sigma, dim=time_dim, mode='constant')
    smoothed = spike_rate.copy(data=smoothed) + mean_spike_rate

    # soft normalize firing rate
    if normalization_scale is not None:
        if isinstance(normalization_scale, str):  # statistic method name
            normalization_scale = getattr(spike_rate, normalization_scale)(dim=dims)
        else:  # scalar or array
            normalization_scale = np.broadcast_to(normalization_scale, unit_shape)
            normalization_scale = xr.DataArray(normalization_scale, coords=unit_coords)
        smoothed /= (normalization_scale + soft_normalize_cut)
    return smoothed
