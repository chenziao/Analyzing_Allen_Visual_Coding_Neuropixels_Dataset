from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
import warnings

from ..analysis.signal import bandpass_filter
from ..utils.quantity_units import as_quantity, as_string

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .data_io import SessionDirectory
    from ..allen_helpers.stimuli import StimulusBlock


def get_layer_channel_groups(
    channel_ids : pd.Index,
    central_channels : dict,
    width : int = 0
) -> xr.DataArray:
    """Get groups of channels around central channel of each layer
    
    Parameters
    ----------
    channel_ids : ArrayLike
        IDs of channels ordered by position.
    central_channels : dict
        Layer structure acronym: ID of central channels in the layer.
    width : int
        Number of channels to the left and right of the central channel.
        Total number of channels in the group is 2 * width + 1.

    Returns
    -------
    xr.DataArray
        DataArray with channel ids for each layer.
        Dimension: layer, channel_index_offset (index offset from central channel).
    """
    channel_groups = []
    for channel in central_channels.values():
        try:
            channel_ids = pd.Index(channel_ids)
            central_idx = channel_ids.get_loc(channel)
        except KeyError:
            warnings.warn(f"Channel '{channel}' not found in channel ids")
            continue
        channels_idx = slice(max(central_idx - width, 0), central_idx + width + 1)
        channel_groups.append(list(channel_ids[channels_idx]))

    channel_groups = xr.DataArray(
        np.array(channel_groups),
        dims=('layer', 'channel_index_offset'),
        coords=dict(
            layer=list(central_channels),
            channel_index_offset=np.arange(-width, width + 1))
    )
    return channel_groups


def get_lfp_channel_groups(
    session_dir : SessionDirectory,
    central_channels : dict,
    probe_id : int | None = None,
    width : int = 0,
    sort_by : str = 'dorsal_ventral_ccf_coordinate',
    **load_lfp_kwargs
) -> tuple[xr.DataArray, xr.DataArray]:
    """Get groups of channels around central channel of each layer
    and average LFP for each group

    Parameters
    ----------
    session_dir : SessionDirectory
        Session directory object.
    central_channels : dict
        Layer structure acronym: ID of central channels in the layer.
    probe_id : int | None
        Probe ID. If None, infer the probe ID from central_channels.
    width : int
        Number of channels to the left and right of the central channel.
        Total number of channels in the group is 2 * width + 1.
    sort_by : str
        Column name in `session.channels` to sort the central channels by.
    **load_lfp_kwargs
        Additional arguments to pass to session_dir.load_lfp.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Average LFP for each group, channel groups.
    """
    all_channels = session_dir.session.channels

    if probe_id is None:
        probe_id = all_channels.loc[next(iter(central_channels.values())), 'probe_id']

    # Get channel groups with layer order sorted by `sort_by` of central channels
    idx = all_channels.loc[central_channels.values(), sort_by].argsort()
    channel_groups = get_layer_channel_groups(
        session_dir.probe_lfp_channels(probe_id).index,
        central_channels, width=width
    ).isel(layer=idx)

    # Load LFP data for unique channels
    unique_channels = list(dict.fromkeys(channel_groups.values.ravel()))  # unique channels
    lfp_array = session_dir.load_lfp(probe_id, channel=unique_channels, **load_lfp_kwargs)

    # Average LFP for each group
    lfp_groups = lfp_array.sel(channel=channel_groups).mean(
        dim='channel_index_offset', keep_attrs=True)
    lfp_groups = lfp_groups.transpose('time', 'layer')
    return lfp_groups, channel_groups


def trial_psd(
    aligned_signal : xr.DataArray,
    tseg : float = 1.,
    df : float | None = None,
    time_dim : str = 'time_from_presentation_onset'
) -> xr.DataArray:
    """Calculate PSD from aligned signal using Welch method
    
    Parameters
    ----------
    aligned_signal : xr.DataArray
        Aligned signal. Attributes must include 'fs'.
    tseg : float
        Segment duration in seconds.
    df : float | None
        Frequency resolution in Hz. If None, use frequency resolution around 1/tseg.
    time_dim : str
        Name of the time dimension in seconds.

    Returns
    -------
    xr.DataArray
        Dimension: dimension 'time_from_presentation_onset' of aligned_signal is replaced by 'frequency'.
        Attributes:
        - 'fs': sampling frequency
        - 'nperseg': number of samples per segment for Welch method
        - 'nfft': number of FFT points for Welch method
        - 'n_dropped': number of samples dropped by Welch method from the last segment
        - 'unit': unit of the signal squared per Hertz
    """
    coords = dict(aligned_signal.coords)
    dims = list(aligned_signal.dims)
    time_axis = dims.index(time_dim)

    fs = aligned_signal.attrs['fs']
    trial_duration = aligned_signal.coords[time_dim][-1] - aligned_signal.coords[time_dim][0]
    nseg = max(np.round(trial_duration / tseg).astype(int).item(), 1)
    # take floor to avoid dropping too many samples by welch method
    nperseg = aligned_signal.coords[time_dim].size // nseg
    n_dropped = aligned_signal.coords[time_dim].size - nperseg * nseg
    if df is None:
        nfft = nperseg
    else:
        nfft = int(np.round(fs / df / 2)) * 2  # ensure nfft is even
    f, pxx = ss.welch(aligned_signal, fs=fs, nperseg=nperseg, nfft=nfft, axis=time_axis)

    del coords[time_dim]
    coords['frequency'] = f
    dims[time_axis] = 'frequency'
    da = xr.DataArray(pxx, coords=coords, dims=dims)
    da.attrs.update(
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        n_dropped=n_dropped,
        unit=as_string(as_quantity(aligned_signal.attrs['unit']) ** 2 / as_quantity('Hz'))
    )
    return da


def bandpass_filter_blocks(
    da : xr.DataArray,
    stimulus_blocks : list[StimulusBlock],
    freq_band : ArrayLike,
    extend_time : float = 0.,
    concat : bool = True,
    time_dim : str = 'time',
    **kwargs
) -> xr.Dataset | list[xr.Dataset]:
    """Bandpass filter blocks of data.

    Parameters
    ----------
    da : xr.DataArray
        Data containing time blocks to filter.
    stimulus_blocks : list[StimulusBlock]
        List of stimulus blocks.
    freq_band : ArrayLike
        Frequency band to filter.
    extend_time : float
        Extend time at the start and end of each block to avoid boundary effect for filtering.
    concat : bool
        Whether to concatenate the filtered blocks into a single DataArray.
    time_dim : str
        Name of the time dimension.
    **kwargs :
        Additional arguments to pass to bandpass_filter.

    Returns
    -------
    xr.Dataset | list[xr.Dataset]
        Filtered blocks.
    """
    filtered_blocks = []
    for stimulus_block in stimulus_blocks:
        block_window = (stimulus_block.block_window[0] - extend_time, stimulus_block.block_window[1] + extend_time)
        filtered_blocks.append(bandpass_filter(
            da.sel({time_dim: slice(*block_window)}), freq_band, **kwargs
        ))
    if concat:
        filtered_blocks = xr.concat(filtered_blocks, dim=time_dim, combine_attrs='override')
        filtered_blocks.attrs['extend_time'] = extend_time
    return filtered_blocks
