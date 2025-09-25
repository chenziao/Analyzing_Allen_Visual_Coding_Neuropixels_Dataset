import numpy as np
import xarray as xr
import scipy.signal as ss
from scipy.ndimage import gaussian_filter

from .utils import array_spacing

from numpy.typing import ArrayLike


def bandpass_filter(
    da : xr.DataArray,
    filt_band : ArrayLike,
    order : int = 4,
    output : str = 'ba',
    time_axis : str = 'time',
    include_filtered : bool = True,
    include_amplitude : bool = False,
    include_phase : bool = False,
    include_analytic : bool = False
) -> xr.Dataset:
    """Filter signal. Get amplitude and phase using Hilbert transform.

    Parameters
    ----------
    da: xarray.DataArray
        Data to filter. DataArray must have time dimension 'time' (ms) or specified by time_axis.
        Attributes can optionally include 'fs' for sampling frequency in Hz.
    filt_band: tuple
        Filter band (low, high) Hz.
    order: int
        Filter order. Default is 4.
    output: str
        Type of output: numerator/denominator ('ba') or second-order sections ('sos').
        Default is 'ba' for backwards compatibility, but 'sos' should be used for
        general-purpose filtering.
    time_axis: str
        Name of the time dimension (ms). Default is None, which will be inferred from the data.
    include_amplitude: bool
        Whether to include amplitude.
    include_phase: bool
        Whether to include phase.
    include_analytic: bool
        Whether to include analytic signal.

    Returns
    -------
    ds: xr.Dataset
        Dataset with filtered data, amplitude, and phase.
        Attributes include 'fs' for sampling frequency in Hz.
    """
    filt_band = tuple(filt_band)
    axis = da.dims.index(time_axis)
    fs = da.attrs.get('fs', 1000. / array_spacing(da.coords[time_axis]))  # Hz

    # Filter
    filt = ss.butter(order, filt_band, btype='bandpass', fs=fs, output=output)
    if output == 'ba':
        filtered = ss.filtfilt(*filt, da.values, axis=axis)
    elif output == 'sos':
        filtered = ss.sosfiltfilt(filt, da.values, axis=axis)
    else:
        raise ValueError(f"Filter type {output} not supported.")
    analytic = ss.hilbert(filtered, axis=axis)

    # Output
    data_vars = {}
    if include_filtered:
        data_vars['filtered'] = da.copy(data=filtered)
    if include_amplitude:
        data_vars['amplitude'] = da.copy(data=np.abs(analytic))
    if include_phase:
        data_vars['phase'] = da.copy(data=np.angle(analytic))
    if include_analytic:
        data_vars['analytic'] = da.copy(data=analytic)
    if not data_vars:
        raise ValueError("No data to include.")

    ds = xr.Dataset(
        data_vars = data_vars,
        attrs = dict(
            fs = fs,
            filt_band = filt_band
        )
    )
    return ds


def compute_csd(
    lfp : xr.DataArray,
    positions : ArrayLike = None,
    sigma_time : float = 1.6,
    sigma_space : float = 40.0,
    padding : tuple = (1, 1),
    time_axis : str = 'time',
    channel_axis : str = 'channel'
) -> xr.DataArray:
    """
    Compute 1D CSD along probe depth, following Allen's style.

    - Replicate padding for δ-Source iCSD boundaries (https://www.sciencedirect.com/science/article/abs/pii/S0165027005004541).
    - Optional 2D Gaussian pre-smoothing across space (channels) and time.
    - Uses 3-point finite difference along depth.

    Parameters
    ----------
    lfp : xr.DataArray
        LFP array. DataArray must have time dimension 'time' (s) and
        channel dimension 'channel' (µm) or specified by time_axis and channel_axis.
        Attributes can optionally include 'fs' for sampling frequency in Hz.
    positions : ArrayLike
        Positions of channels in µm (must be roughly uniform)
        If not provided, assume the channel dimension is the position.
    sigma_time : float
        Temporal Gaussian sigma in ms (default 1.6 ms ~ 2 samples at 1250 Hz).
    sigma_space : float
        Spatial Gaussian sigma in µm (default 40 µm, 1x spacing between channels).
    time_axis : str
        Name of the time dimension. Default is 'time'.
    channel_axis : str
        Name of the channel dimension. Default is 'channel'.

    Returns
    -------
    da: xr.DataArray
        DataArray with CSD.
        Attributes include 'fs', 'sigma_time', and 'sigma_space'.
    """
    c_axis = lfp.dims.index(channel_axis)
    t_axis = lfp.dims.index(time_axis)
    if positions is None:
        positions = lfp.coords[channel_axis].values
    dc = array_spacing(positions)  # µm
    fs = lfp.attrs.get('fs', 1. / array_spacing(lfp.coords[time_axis]))  # Hz

    # Optional 2D pre-smoothing
    if sigma_time or sigma_space:
        sigma_c = (sigma_space / dc) if sigma_space else 0.0
        sigma_t = (sigma_time / 1000.0) * fs if sigma_time else 0.0
        sigma = [0] * lfp.ndim
        sigma[c_axis] = sigma_c
        sigma[t_axis] = sigma_t
        x = gaussian_filter(lfp.values, sigma=sigma, mode='nearest')
    else:
        x = lfp.values

    x = np.moveaxis(x, c_axis, 0) # move channel axis to first dimension

    # Replicate padding in channel dimension
    if any(padding):
        pad = [(0, 0)] * lfp.ndim
        pad[0] = padding
        xpad = np.pad(x, pad, mode='edge')
    else:
        xpad = x

    # 3-point 2nd difference
    csd = (xpad[2:] - 2 * xpad[1:-1] + xpad[:-2]) / (dc ** 2)
    # Move channel axis back to original position
    csd = np.moveaxis(csd, 0, c_axis)

    # Update coordinates to exclude boundary channels
    coords = dict(lfp.coords)
    channel_slice = slice(None if padding[0] else 1, None if padding[1] else -1)
    coords[channel_axis] = coords[channel_axis][channel_slice]

    # Create DataArray
    da = xr.DataArray(data=csd, dims=lfp.dims, coords=coords)
    da.attrs.update(dict(lfp.attrs) | dict(
        fs=fs,
        sigma_time=sigma_time,
        sigma_space=sigma_space
    ))
    return da
