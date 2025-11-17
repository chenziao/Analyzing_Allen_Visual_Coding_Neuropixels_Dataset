from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss

from ..analysis.signal import bandpass_filter
from ..utils.quantity_units import as_quantity, as_string

from typing import Sequence
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .data_io import SessionDirectory
    from ..allen_helpers.stimuli import StimulusBlock


def get_lfp_channel_groups(
    session_dir : SessionDirectory,
    central_channels : dict,
    probe_id : int | None = None,
    width : int = 0,
    sort_by : str = 'probe_vertical_position',
    descending : bool = True,
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
        Default is to sort by vertical position in descending order (superficial to deep).
    descending : bool
        Whether to sort the channels in descending order.
    **load_lfp_kwargs
        Additional arguments to pass to session_dir.load_lfp.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        Average LFP for each group, channel groups.
    """
    if probe_id is None:
        all_channels = session_dir.session.channels
        for channel in central_channels.values():
            if channel in all_channels.index:
                probe_id = all_channels.loc[channel, 'probe_id']
                break
        else:
            raise ValueError("No probe ID found for any of the central channels")

    from .location import argsort_channels, get_layer_channel_groups

    # Get channel groups with layer order sorted by `sort_by` of central channels
    probe_channels = session_dir.probe_lfp_channels(probe_id)
    idx = argsort_channels(central_channels.values(),
        channels_df=probe_channels, sort_by=sort_by, descending=descending)
    channel_groups = get_layer_channel_groups(
        probe_channels.index, central_channels, width=width).isel(layer=idx)

    # Load LFP data for unique channels
    unique_channels = list(dict.fromkeys(channel_groups.values.ravel()))  # unique channels
    lfp_array = session_dir.load_lfp(probe_id=probe_id, channel=unique_channels, **load_lfp_kwargs)

    # Average LFP for each group
    lfp_groups = lfp_array.sel(channel=channel_groups).mean(
        dim='channel_index_offset', keep_attrs=True)
    lfp_groups = lfp_groups.transpose('time', 'layer')
    return lfp_groups, channel_groups


def trial_psd(
    aligned_signal : xr.DataArray,
    tseg : float = 1.,
    df : float | None = None,
    poverlap : float = 0.5,
    window : str = 'hann',
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
    poverlap : float
        Overlap proportion between segments. Must be between 0 and 1.
    window : str
        Window function to use for Welch method. See scipy.signal.welch for available options.
    time_dim : str
        Name of the time dimension in seconds.

    Returns
    -------
    xr.DataArray
        Dimension: dimension 'time_from_presentation_onset' of aligned_signal is replaced by 'frequency'.
        Attributes:
        - 'fs': sampling frequency
        - 'nperseg': number of samples per segment for Welch method
        - 'noverlap': number of overlap samples between segments for Welch method
        - 'nseg': number of segments for Welch method
        - 'nfft': number of FFT points for Welch method
        - 'n_dropped': number of samples dropped by Welch method from the last segment
        - 'unit': unit of the signal squared per Hertz
    """
    coords = dict(aligned_signal.coords)
    dims = list(aligned_signal.dims)
    time_axis = dims.index(time_dim)
    assert poverlap >= 0 and poverlap < 1, "poverlap must be between 0 and 1"

    # determine parameters for Welch method
    fs = aligned_signal.attrs['fs']
    N = coords[time_dim].size  # total number of samples per trial
    nseg = max(int(np.round((N / fs / tseg - poverlap) / (1 - poverlap))), 1)
    # take floor to avoid dropping too many samples by welch method
    nperseg = int(np.floor(N / (nseg * (1 - poverlap) + poverlap)))
    # calculate actual noverlap and number of segments
    noverlap = int(np.floor(poverlap * nperseg))
    nseg = (N - noverlap) // (nperseg - noverlap)
    n_dropped = N - (nseg * (nperseg - noverlap) + noverlap)
    if df is None:
        nfft = nperseg
    else:
        nfft = int(np.round(fs / df / 2)) * 2  # ensure nfft is even

    # calculate PSD using Welch method
    f, pxx = ss.welch(aligned_signal, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=time_axis)

    del coords[time_dim]
    coords['frequency'] = f
    dims[time_axis] = 'frequency'
    da = xr.DataArray(pxx, coords=coords, dims=dims)
    da.attrs.update(
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nseg=nseg,
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


def average_psd_across_sessions(
    psd_ds: Sequence[xr.Dataset] | Sequence[xr.DataArray],
    fs: float | None = None,
    nfft: int | None = None
) -> tuple[xr.Dataset, xr.Dataset] | tuple[xr.DataArray, xr.DataArray]:
    """Average PSD data arrays or datasets in decibels across sessions.
    
    Parameters
    ----------
    psd_ds: Sequence[xr.Dataset] | Sequence[xr.DataArray]
        PSD datasets or data arrays to average across sessions. Must have same dimensions.
    fs: float | None
        Common sampling frequency. If None, use the median sampling frequency of the sessions.
    nfft: int | None
        Common number of FFT points. If None, use the median number of FFT points of the sessions.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset] | tuple[xr.DataArray, xr.DataArray]:
        Average PSD across sessions, concatenated PSD datasets or data arrays.

    Note: Missing values are skipped when averaging.
    """
    # get sessions with stimulus
    psd_ds = list(psd_ds)
    n_seesions = len(psd_ds)
    if not n_seesions:
        raise ValueError("No PSD sessions provided")

    # get common frequency array and attributes
    attrs = dict(psd_ds[0].attrs)
    attrs.pop('session_id', None)
    if fs is None:
        fs = np.median([ds.attrs['fs'] for ds in psd_ds])
    if nfft is None:
        nfft = np.median([ds.attrs['nfft'] for ds in psd_ds])
    frequency = np.fft.rfftfreq(nfft, d=1/fs)
    attrs.update(fs=fs, nfft=nfft)

    # gether and concatenate psd of stimulus for each session
    session_ids = [ds.attrs['session_id'] for ds in psd_ds]
    if isinstance(psd_ds[0], xr.Dataset):  # drop data variable `channel_groups` (not PSD data)
        psd_ds = [ds.drop_vars('channel_groups') for ds in psd_ds]
    psd_ds = [ds.assign_coords({'frequency': frequency}) for ds in psd_ds]  # use common frequency
    psd_ds = xr.concat(psd_ds, dim=pd.Index(session_ids, name='session_id'), combine_attrs='drop_conflicts')
    psd_ds = psd_ds.assign_attrs(attrs)

    # average across sessions in decibels
    psd_avg = (10 * np.log10(psd_ds)).mean(dim='session_id', skipna=True)
    psd_avg = 10 ** (psd_avg / 10) # convert back to linear scale
    psd_avg = psd_avg.assign_attrs(attrs, n_sessions=n_seesions)
    return psd_avg, psd_ds
