from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss

from ..analysis.signal import bandpass_filter
from ..utils.quantity_units import as_quantity, as_string

from typing import Sequence, Callable
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .data_io import SessionDirectory
    from ..allen_helpers.stimuli import StimulusBlock
    from ..analysis.spectrum import FOOOF


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

    Note: NaN values are replaced with zeros and the PSD is scaled by 1 / (1 - p_nan) to avoid bias due to NaN values.
    """
    coords = dict(aligned_signal.coords)
    dims = list(aligned_signal.dims)
    time_axis = dims.index(time_dim)
    assert poverlap >= 0 and poverlap < 1, "poverlap must be between 0 and 1"
    unit = aligned_signal.attrs['unit']

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

    # check for NaN values and replace with zeros if any
    is_nan = np.isnan(aligned_signal.values)
    n_nan = is_nan.sum()
    if n_nan:
        p_nan = n_nan / aligned_signal.size
        aligned_signal = np.where(is_nan, 0., aligned_signal)
        print(f"Warning: {100 * p_nan:.2f}% NaN values found in signal. Replace with zeros.")

    # calculate PSD using Welch method
    f, pxx = ss.welch(aligned_signal, fs=fs, window=window,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=time_axis)

    if n_nan:
        pxx /= (1 - p_nan)  # scale PSD by 1 / (1 - p_nan) to avoid bias due to NaN values

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
        unit=as_string(as_quantity(unit) ** 2 / as_quantity('Hz'))
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


def filter_conditions(da: xr.DataArray, condition_filters: dict[str, str | Callable] | None = None) -> xr.DataArray:
    """Filter data array by drifting grating conditions.
    
    Parameters
    ----------
    da: xr.DataArray
        Data array to filter. Dimensions include drifting grating conditions.
    condition_filters: dict[str, str | Callable] | None
        Dictionary of condition filters. Keys are dimension names, 
        values are filter functions or strings of lambda functions.

    Returns
    -------
    xr.DataArray
        Filtered data array.
    """
    if condition_filters is None:
        from toolkit.pipeline.global_settings import GLOBAL_SETTINGS
        condition_filters = GLOBAL_SETTINGS['condition_filters']
    filtered_sel = {}
    for dim in condition_filters.keys() & da.coords.keys():
        filter_func = condition_filters[dim]
        if isinstance(filter_func, str):
            filter_func = eval(filter_func)
        filtered_sel[dim] = filter_func(da.coords[dim])
    return da.sel(filtered_sel)


def fit_fooof_and_get_bands(
    psd_das : xr.Dataset | dict[str, xr.DataArray],
    fooof_params : dict,
    freq_band_kwargs : dict,
    wave_band_limit : xr.DataArray,
    wave_band_width_limit : xr.DataArray
) -> tuple[dict[str, dict[str, FOOOF]], xr.Dataset]:
    """Fit FOOOF and get frequency bands.
    
    Parameters
    ----------
    psd_das : xr.Dataset | dict[str, xr.DataArray]
        Dataset or dictionary of data arrays of PSD for each stimulus.
    fooof_params : dict
        Parameters for `toolkit.analysis.spectrum.fit_fooof`.
    freq_band_kwargs : dict
        Keyword arguments for `toolkit.analysis.spectrum.get_fooof_freq_band`.
    wave_band_limit : xr.DataArray
        Wave band limit. Must have 'wave_band' and 'bound' dimensions.
    wave_band_width_limit : xr.DataArray
        Wave band width limit. Must have 'wave_band' dimension.

    Returns
    -------
    fooof_objs : dict[str, dict[str, FOOOF]]
        FOOOF objects for each stimulus and layer.
    bands_ds : xr.Dataset
        Dataset of frequency bands detected by FOOOF.
        Dimensions: 'stimulus', 'layer', 'wave_band', 'bound'.
        Data variables:
            bands : frequency band in Hz.
            peaks : Top N peaks of gaussian parameters.
            center_freq : Center frequency of the top N peaks in Hz.
            wave_band_limit, wave_band_width_limit : Wave band limit and width limit parameters used.
        Note: Frequency bands undetected by FOOOF are filled with NaN.
    """
    from ..analysis.spectrum import fit_fooof, get_fooof_freq_band
    if isinstance(psd_das, xr.Dataset):
        psd_das = dict(psd_das.data_vars)
    stimulus_names = list(psd_das)
    layers = next(iter(psd_das.values())).coords['layer'].values

    bands = np.full((len(stimulus_names), layers.size, wave_band_limit.wave_band.size, 2), np.nan)
    peaks = np.full(bands.shape[:-1] + (freq_band_kwargs['top_n_peaks'], ), np.nan)
    center_freq = peaks.copy()

    fooof_objs = {}
    for i, (stim, psd_avg) in enumerate(psd_das.items()):
        fooof_objs[stim] = {}
        for j, layer in enumerate(layers):
            # fit fooof
            fooof_results = fit_fooof(psd_avg.sel(layer=layer), **fooof_params)
            gaussian_params = fooof_results[0].gaussian_params
            fooof_objs[stim][layer] = fooof_results[1]

            # get frequency bands
            for k, wave_band in enumerate(wave_band_limit.wave_band):
                band, peak_inds = get_fooof_freq_band(
                    gaussian_params=gaussian_params,
                    freq_range=wave_band_limit.sel(wave_band=wave_band).values,
                    width_limit=wave_band_width_limit.sel(wave_band=wave_band).values,
                    **freq_band_kwargs
                )

                bands[i, j, k] = band
                peaks[i, j, k, :peak_inds.size] = gaussian_params[peak_inds, 1]
                center_freq[i, j, k, :peak_inds.size] = gaussian_params[peak_inds, 0]

    coords = dict(stimulus=stimulus_names, layer=layers, wave_band=wave_band_limit.coords['wave_band'])
    bands = xr.DataArray(data=bands, coords=coords | dict(bound=wave_band_limit.coords['bound']))
    peaks = xr.DataArray(data=peaks, coords=coords | dict(peak_rank=range(peaks.shape[-1])))
    center_freq = peaks.copy(data=center_freq)

    bands_ds = xr.Dataset(dict(
        bands=bands, peaks=peaks, center_freq=center_freq,
        wave_band_limit=wave_band_limit,
        wave_band_width_limit=wave_band_width_limit
    ))
    bands_ds.attrs.update(fooof_results[2] | fooof_params | freq_band_kwargs)
    return fooof_objs, bands_ds


def layer_condition_band_power(
    cond_psd_da : xr.DataArray,
    bands_da : xr.DataArray,
    wave_band_limit : xr.DataArray,
    fixed_condition_types : Sequence[str],
    condition_wave_band : str = 'beta'
) -> tuple[xr.DataArray, xr.Dataset]:
    """Get band power in drifting grating conditions for each layer.
    
    Parameters
    ----------
    cond_psd_da : xr.DataArray
        PSD data array. Dimensions: layer, frequency, *condition_types.
    bands_da : xr.DataArray
        Bands data array. Dimensions: layer, wave_band, bound.
    wave_band_limit : xr.DataArray
        Wave band limit.
    fixed_condition_types : Sequence[str]
        Fixed condition types.
    condition_wave_band : str
        Wave band of which the power is calculated.

    Returns
    -------
    cond_band_power : xr.DataArray
        Band power in drifting grating conditions for each layer.
        Dimensions: layer, *condition_types.
    layer_bands_ds : xr.Dataset
        Dataset of frequency bands detected by FOOOF for each layer.
        Data variables:
            bands : frequency band in Hz. Default to `wave_band_limit` if not detected by FOOOF.
            detected : whether the frequency band is detected by FOOOF.
    """
    cond_band_power = {}
    layer_bands = []
    layer_detected = []
    layers = bands_da.coords['layer'].values
    for layer in layers:
        band = bands_da.sel(layer=layer, wave_band=condition_wave_band)
        detected = not np.isnan(band).any()
        if not detected:  # use frequency range of interest if band is not detected by fooof
            band = wave_band_limit.sel(wave_band=condition_wave_band)
        layer_detected.append(detected)
        layer_bands.append(band.values)

        cond_psd = cond_psd_da.sel(layer=layer)
        power = cond_psd.sel(frequency=slice(*band)).integrate('frequency').mean(dim=fixed_condition_types)
        unit = as_string(as_quantity(cond_psd.attrs['unit']) * as_quantity('Hz'))
        cond_band_power[layer] = power

    cond_band_power = xr.concat(cond_band_power.values(),
        dim=pd.Index(cond_band_power, name='layer'), combine_attrs='override')
    cond_band_power.attrs.update(cond_psd_da.attrs | dict(unit=unit, wave_band=condition_wave_band))
    layer_bands_ds = xr.Dataset(
        data_vars=dict(
            bands=(('layer', 'bound'), layer_bands),
            detected=('layer', layer_detected)
        ),
        coords=dict(layer=layers, bound=bands_da.coords['bound'])
    ).assign_attrs(wave_band=condition_wave_band)
    return cond_band_power, layer_bands_ds


def get_band_with_highest_peak(bands_ds: xr.Dataset, dim: str | Sequence[str] | None = None) -> xr.DataArray | None:
    """Get the band with the highest peak along a dimension/dimensions.

    Parameters
    ----------
    bands_ds : xr.Dataset
        Dataset of bands including 'bands' and 'peaks' data variables.
        Allow only a single dimension of 'bands' except 'bound' and of 'peaks' except 'peak_rank'.
    dim : str | Sequence[str] | None
        Dimension(s) to get the band from.
        If not provided, along all dimensions that are not 'bound' or 'peak_rank'.

    Returns
    -------
    band : xr.DataArray
        Band with the highest peak. None if no peak is detected.
    """
    peaks = bands_ds.peaks.sel(peak_rank=0)
    if np.isnan(peaks).all():
        return None
    if dim is None:
        dim = (d for d in bands_ds.dims if d not in {'bound', 'peak_rank'})
    elif isinstance(dim, str):
        dim = [dim]
    return bands_ds.bands.isel(peaks.argmax(dim=list(dim)))


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
    psd_ds = [ds.assign_coords({'frequency': frequency}) for ds in psd_ds]  # use common frequency
    psd_ds = xr.concat(psd_ds, dim=pd.Index(session_ids, name='session_id'), combine_attrs='drop_conflicts')
    psd_ds = psd_ds.assign_attrs(attrs)

    # average across sessions in decibels
    from ..analysis.spectrum import average_psd_in_decibels
    psd_avg = average_psd_in_decibels(psd_ds, dim='session_id').assign_attrs(attrs, n_sessions=n_seesions)
    return psd_avg, psd_ds
