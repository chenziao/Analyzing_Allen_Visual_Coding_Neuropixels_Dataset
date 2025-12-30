from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from ..analysis.spectrum import _get_psd_freq_range
from .format import UNIT

from typing import Any, Literal, Sequence
from matplotlib.pyplot import Figure, Axes
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from fooof import FOOOF
    from fooof.data import FOOOFResults


def plot_probe_channel_positions(channels : pd.DataFrame, ax : Axes | None = None):
    """Plot probe channel positions.

    Parameters
    ----------
    channels: pandas DataFrame returned by session.channels. It should have columns
        'probe_vertical_position' and 'probe_horizontal_position' and 'probe_channel_number'.
    ax: matplotlib Axes object
        If None, a new figure and axes will be created.

    Returns
    -------
    ax: matplotlib Axes object
        The axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 3))
    ax.plot(channels['probe_vertical_position'], channels['probe_horizontal_position'], 'b.')
    for _, row in channels.iterrows():
        ax.annotate(str(row['probe_channel_number']),
                    (row['probe_vertical_position'], row['probe_horizontal_position']),
                    fontsize=6, color='k')
    ax.set_xlabel(f'Vertical ({UNIT.um})')
    ax.set_ylabel(f'Horizontal ({UNIT.um})')
    ax.set_title('Channel positions')
    ax.set_ylim(0, 70)  # width of the probe
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    return ax


def plot_channel_signal_array(
    time : ArrayLike,
    channel_positions : pd.Series,
    signal : ArrayLike,
    central_channels : dict[str, int] | None = None,
    coordinates : ArrayLike | None = None,
    coordinates_label : str = 'Dorsal-Ventral CCF',
    coordinates_tick_step : int = 4,
    ax : Axes | None = None,
    pcolormesh_kwargs : dict = {},
    xlabel : str = f'Time ({UNIT.s})',
    clabel : str = 'CSD',
    c_unit : str = None,
):
    """Plot signal array along channels and time.
    Show Dorsal-Ventral CCF coordinates and layer labels along the channel axis (y-axis).

    Parameters
    ----------
    time : ArrayLike
        Time.
    channel_positions : pd.Series
        Channel vertical positions. Index is the channel number.
    signal : ArrayLike
        Signal array. Shape: (channel, time)
    central_channels : dict[str, int] | None
        Dictionary of central channels for each layer {layer acronym: channel id}.
        If specified, show the layer labels on y-axis.
    coordinates : ArrayLike | None
        Coordinates of the channels. If specified, show the coordinates on y-axis.
    coordinates_label : str
        Label for the coordinates axis. Default is 'Dorsal-Ventral CCF'.
    coordinates_tick_step : int
        Step for the coordinates tick labels (number of channels between ticks).
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
    pcolormesh_kwargs : dict
        Keyword arguments for pcolormesh.
    xlabel : str
        Label for the x-axis.
    clabel : str
        Label for the colorbar.
    c_unit : str | None
        Unit for the colorbar. Set to empty string to not show the unit.
        If not specified, use the unit from the signal attributes if available.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    # Heatmap
    pcm = ax.pcolormesh(time, channel_positions, signal, **pcolormesh_kwargs)

    # Colorbar
    if c_unit is None and isinstance(signal, xr.DataArray):
        c_unit = signal.attrs.get('unit', '')
    if c_unit:
        clabel += f' ({UNIT[c_unit]})'
    plt.colorbar(mappable=pcm, ax=ax, label=clabel)

    ax.set_xlabel(xlabel)

    # Layer labels
    if central_channels is not None:
        ax.set_ylabel('Layer', labelpad=0)
        ax.set_yticks(channel_positions[central_channels.values()])
        ax.set_yticklabels(central_channels.keys())

    # Show ccf coordinates
    if coordinates is not None:
        if central_channels is not None:
            ax.spines['left'].set_position(('outward', 50))  # Move the labels outward
            ax2 = ax.twinx()  # Add a second y-axis with CCF labels
            ax2.set_ylim(ax.get_ylim())
            ax2.yaxis.set_ticks_position('left')
            ax2.yaxis.set_label_position('left')
        else:
            ax2 = ax

        ax2.set_ylabel(f'{coordinates_label} ({UNIT.um})', labelpad=0)

        # Choose a subset for labeling every 4th channel
        subset_idx = np.arange(0, len(channel_positions), coordinates_tick_step)
        subset_labels = map(str, np.round(np.asarray(coordinates)[subset_idx]).astype(int))
        ax2.set_yticks(channel_positions.iloc[subset_idx])
        ax2.set_yticklabels(subset_labels)

    return ax


def plot_channel_psd(
    psd_da : xr.DataArray,
    channel_dim : str = 'channel',
    freq_range : ArrayLike | float = (),
    power_scale : Literal['dB', 'log', 'linear'] = 'dB',
    log_frequency : bool = False,
    cmap : str = 'plasma',
    channel_title : str | None = None,
    ax : Axes | None = None
):
    """Plot PSD at available channels.
    
    Parameters
    ----------
    psd_da : xr.DataArray
        PSD data array. Dimension: 'frequency', channel_dim.
    channel_dim : str
        Dimension name for channels.
    freq_range : ArrayLike | float
        Frequency range to plot. If float, use (0, freq_range) Hz.
        If ArrayLike, use (freq_range[0], freq_range[1]) Hz.
        If ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    power_scale : Literal['dB', 'log', 'linear']
        Power scale. 'log' or 'linear' for using original unit, 'dB' for converting to decibels.
    log_frequency : bool
        Whether using log frequency scale.
    cmap : str
        Colormap for channels.
    channel_title : str | None
        Title for the channel legend.
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
    
    Returns
    -------
    ax : Axes
        Axes object with the plot.
    """
    freq_range = _get_psd_freq_range(freq_range, frequencies=psd_da.frequency, log_frequency=log_frequency)
    psd_plt_da = psd_da.sel(frequency=slice(*freq_range))
    psd_plt_da = psd_plt_da.transpose('frequency', channel_dim)

    if ax is None:
        _, ax = plt.subplots(1, 1)

    if channel_title is None:
        channel_title = channel_dim.title()
    channels = psd_da.coords[channel_dim].values
    colors = plt.get_cmap(cmap, channels.size)(np.linspace(0, 1, channels.size))
    ax.set_prop_cycle(plt.cycler('color', colors))

    if power_scale.lower() == 'db':
        psd_plt_da = 10 * np.log10(psd_plt_da)
        unit = 'dB/Hz'
    else:
        unit = UNIT[psd_da.attrs.get('unit', '')]

    ax.plot(psd_plt_da.frequency, psd_plt_da.values, label=channels)

    if power_scale.lower() == 'log':
        ax.set_yscale('log')
    if log_frequency:
        ax.set_xscale('log')
    ax.set_xlim(freq_range)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'PSD ({unit})')
    ax.legend(loc='upper right', framealpha=0.2, title=channel_title)
    return ax


def plot_fooof_quick(
    fooof : FOOOF,
    plot_peaks : Literal['shade', 'dot', 'outline', 'line'] | None = 'dot',
    freq_range=(),
    plt_log=False,
    ax=None,
    **kwargs
):
    """Helper function for plotting FOOOF results using built-in function FOOOF.plot().
    This function constraints the frequency range to avoid errors.
    
    Parameters
    ----------
    fooof : FOOOF
        FOOOF object.
    plot_peaks : Literal['shade', 'dot', 'outline', 'line'] | None
        Whether to plot the peaks.
    freq_range : ArrayLike | float
        Frequency range to plot. If float, use (0, freq_range) Hz.
        If ArrayLike, use (freq_range[0], freq_range[1]) Hz.
        If ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    plt_log : bool
        Whether to plot the frequency axis in log scale.
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
    **kwargs : dict
        Keyword arguments for FOOOF.plot().

    Returns
    -------
    ax : Axes
        Axes object with the plot.
    """
    freq_range = _get_psd_freq_range(freq_range, frequencies=fooof.freqs, log_frequency=plt_log)
    fooof.plot(plot_peaks=plot_peaks, freq_range=freq_range, plt_log=plt_log, ax=ax, **kwargs)
    if ax is None:
        ax = plt.gca()
    return ax


def plot_fooof(psd_da : xr.DataArray | ArrayLike,
    f : ArrayLike | None = None,
    fooof_result : FOOOFResults | None = None,
    freq_range : ArrayLike | float = (),
    power_scale : Literal['dB', 'log', 'linear'] = 'dB',
    log_frequency : bool = False,
    ax : Axes | None = None
):
    """Plot FOOOF results in customized style.
    
    Parameters
    ----------
    psd_da : xr.DataArray | ArrayLike
        Power spectral density array. Dimension should be 'frequency'.
    f : ArrayLike | None
        Frequency array. Used only when psd_da is not a DataArray. Otherwise ignored.
    fooof_result : FOOOFResults | None
        FOOOF results. If None, fit FOOOF with default parameters.
    freq_range : ArrayLike | float
        Frequency range to plot. If float, use (0, freq_range) Hz.
        If ArrayLike, use (freq_range[0], freq_range[1]) Hz.
        If ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    power_scale : Literal['dB', 'log', 'linear']
        Power scale. 'log' or 'linear' for using original unit, 'dB' for converting to decibels.
    log_frequency : bool
        Whether using log frequency scale.
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
        
    Returns
    -------
    ax : Axes
        Axes object with the plot.
    """
    if isinstance(psd_da, xr.DataArray):
        pxx = psd_da.values
        f = psd_da.coords['frequency'].values
        unit = psd_da.attrs.get('unit', '')
    else:
        pxx = np.asarray(psd_da)
        f = np.asarray(f)
        unit = ''

    if fooof_result is None:
        from ..analysis.spectrum import fit_fooof
        fooof_result = fit_fooof(pxx, f=f)[0]

    from fooof.sim.gen import gen_model
    # get full and aperiodic fits
    full_fit, _, ap_fit = gen_model(f[1:], fooof_result.aperiodic_params,
                                    fooof_result.gaussian_params, return_components=True)
    full_fit = np.insert(10 ** full_fit, 0, pxx[0])  # insert DC component
    ap_fit = np.insert(10 ** ap_fit, 0, pxx[0])  # insert DC component

    if ax is None:
        _, ax = plt.subplots(1, 1)

    freq_range = _get_psd_freq_range(freq_range, frequencies=f, log_frequency=log_frequency)
    f_idx = (f >= freq_range[0]) & (f <= freq_range[1])
    f, pxx = f[f_idx], pxx[f_idx]
    full_fit, ap_fit = full_fit[f_idx], ap_fit[f_idx]

    if power_scale.lower() == 'db':
        pxx, full_fit, ap_fit = [10 * np.log10(x) for x in [pxx, full_fit, ap_fit]]
        unit = 'dB/Hz'

    ax.plot(f, pxx, 'k', label='Original')
    ax.plot(f, full_fit, 'r', label='Full model fit')
    ax.plot(f, ap_fit, 'b--', label='Aperiodic fit')

    if power_scale.lower() == 'log':
        ax.set_yscale('log')
    if log_frequency:
        ax.set_xscale('log')
    ax.set_xlim(freq_range)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'PSD ({unit})')
    ax.legend(loc='upper right', frameon=False)
    return ax


def plot_freq_band(
    bands : Sequence[tuple[float, float] | None] | tuple[float, float] | None,
    band_labels : Sequence[str] | str | None = None,
    band_colors_map : dict[str, Any] | None = None,
    alpha : float = 0.1,
    ax : Axes | None = None,
    **kwargs
):
    """Plot the frequency band as vertical shaded regions.
    
    Parameters
    ----------
    bands : Sequence[tuple[float, float]] | tuple[float, float]
        Frequency bands.
    band_labels : Sequence[str] | str | None
        Labels for the bands. If None, no labels will be shown.
    band_colors_map : dict[str, Any] | None
        Map of band labels to colors.
    alpha : float
        Alpha value for the shaded regions.
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
    **kwargs : dict
        Keyword arguments for ax.axvspan().
    """
    if band_colors_map is None:
        from .colors import BAND_COLORS
        band_colors_map = BAND_COLORS  # default color map

    if ax is None:
        _, ax = plt.subplots(1, 1)

    bands = np.asarray(bands, dtype=object)
    if not np.any(bands == None):
        bands = np.atleast_2d(bands.astype(float))

    if band_labels is None:
        band_labels = None
    band_labels = np.broadcast_to(np.asarray(band_labels, dtype=object), bands.shape[0])

    for band, label in zip(bands, band_labels):
        if band is None:
            continue
        ax.axvspan(*band[:2], label=label, color=band_colors_map[label],
            alpha=alpha, linestyle='none', **kwargs)

    if any(band_labels):
        ax.legend()  # Update legend for the shaded regions
    return ax


def plot_layer_condition_band_power(
    cond_band_power : xr.DataArray,
    layer_bands_ds : xr.Dataset,
    x_cond : str,
    y_cond : str,
    figsize : tuple[float, float] = (4.8, 3.0)
) -> tuple[Figure, np.ndarray]:
    """Plot heatmap of band power in conditions for each layer.
    
    Parameters
    ----------
    cond_band_power : xr.DataArray
        Band power in drifting grating conditions for each layer.
        Dimensions: layer, *condition_types.
    layer_bands_ds : xr.Dataset
        Dataset of frequency bands detected by FOOOF for each layer.
        Data variables:
            bands : frequency band in Hz. Default to `wave_band_limit` if not detected by FOOOF.
            detected : whether the frequency band is detected by FOOOF.
    x_cond, y_cond : str
        Condition types for the x-axis and y-axis.
    figsize : tuple[float, float]
        Figure subplot size.

    Returns
    -------
    fig : Figure
        Figure object.
    axs : np.ndarray
        Array of Axes objects.
    """
    from ..allen_helpers.stimuli import COND_LABEL

    cond_power_db = 10 * np.log10(cond_band_power)
    vmin, vmax = cond_power_db.min(), cond_power_db.max()
    layers = cond_power_db.layer.values
    fig, axs = plt.subplots(layers.size, 1, squeeze=False, figsize=(figsize[0], figsize[1] * layers.size))
    for layer, ax in zip(layers, axs.ravel()):
        cpower = cond_power_db.sel(layer=layer).transpose(y_cond, x_cond)
        label = layer_bands_ds.attrs['wave_band'].title() + ' power (dB)'
        band = layer_bands_ds.bands.sel(layer=layer).values

        x = cpower.coords[x_cond].values
        y = cpower.coords[y_cond].values
        cpower = cpower.assign_coords({x_cond: range(x.size), y_cond: range(y.size)})

        im = ax.imshow(cpower, origin='lower', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, label=label, pad=0.03)
        ax.set_xticks(cpower.coords[x_cond], labels=map('{:g}'.format, x))
        ax.set_yticks(cpower.coords[y_cond], labels=map('{:g}'.format, y))
        ax.set_xlabel(COND_LABEL[x_cond])
        ax.set_ylabel(COND_LABEL[y_cond])
        title = f'Layer {layer}, {band[0]:.1f} - {band[1]:.1f} Hz'
        if not layer_bands_ds.detected.sel(layer=layer).item():
            title += ' (undetected)'
        ax.set_title(title)
    fig.tight_layout()
    return fig, axs


def plot_optotag_evoke_ratio(
    optotag_df : pd.DataFrame,
    min_rate : float = 1.,
    evoked_ratio_threshold : float | None = None,
    ax : Axes | None = None
):
    log_gap = 2 ** 0.5  # gap for log scale plot axis limit
    max_rate = max(optotag_df['opto_evoked_rate'].max(), optotag_df['opto_baseline_rate'].max()) + min_rate
    rate_limit = np.array([min_rate / log_gap, max_rate * log_gap])
    baseline_rate_adjusted = optotag_df['opto_baseline_rate'] + min_rate
    evoked_rate_adjusted = optotag_df['opto_evoked_rate'] + min_rate
    evoke_significant = optotag_df['evoke_significant'].values
    n_units = len(optotag_df)
    n_sig = sum(evoke_significant)

    _, ax = plt.subplots(1, 1)
    if evoked_ratio_threshold is not None:
        n_pos = sum(optotag_df['evoke_positive'])
        ax.plot(rate_limit, rate_limit * evoked_ratio_threshold, '--r',
            label=f'Evoked ratio threshold: {evoked_ratio_threshold:g}\nPositive units: {n_pos:d} / {n_units:d}')
    ax.plot(rate_limit, rate_limit, '--k', label='Identity line')
    ax.plot(baseline_rate_adjusted[~evoke_significant], evoked_rate_adjusted[~evoke_significant],
        color='b', linestyle='none', marker='.')
    ax.plot(baseline_rate_adjusted[evoke_significant], evoked_rate_adjusted[evoke_significant],
        color='b', linestyle='none', marker='o', markerfacecolor='none', label=f'Significant units: {n_sig:d} / {n_units:d}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(rate_limit)
    ax.set_ylim(rate_limit)
    ax.set_xlabel('Baseline rate (Hz)')
    ax.set_ylabel('Evoked rate (Hz)')
    ax.legend(loc='lower right', framealpha=0.2)
    return ax


def plot_optotag_units(
    optotag_df : pd.DataFrame,
    evoked_ratio_threshold : float | None = None,
    ttest_alpha : float | None = None,
    spike_width_range : tuple[float, float] | None = None,
    marker_size : float = 3,
    ax : Axes | None = None
):
    """Plot scatter plot of optotag units with evoked ratio and spike width thresholds
    
    Parameters
    ----------
    optotag_df : pd.DataFrame
        Dataframe containing the units information and optotagging results.
    evoked_ratio_threshold : float | None
        Threshold for evoked ratio. If None, no threshold is applied.
    ttest_alpha : float | None
        Alpha for t-test. If None, no t-test is performed.
    spike_width_range : tuple[float, float] | None
        Range for spike width. If None, no spike width range limit is applied.
    marker_size : float
        Marker size.
    ax : Axes | None
        Axes to plot on. If None, a new figure and axes will be created.
        
    Returns
    -------
    ax : Axes
        Axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if ttest_alpha is None:
        ttest_alpha = 0.5
    if spike_width_range is None:
        spike_width_range = (0., np.inf)
    alpha_size = -marker_size * np.log2(ttest_alpha)
    if 'opto_p_value' in optotag_df.columns:
        marker_sizes = marker_size * np.clip(-np.log2(optotag_df['opto_p_value']), 1, 30)
    else:
        marker_sizes = alpha_size
    positive_units = optotag_df['optotag_positive']
    colors = np.array(['b', 'r'])[positive_units.astype(int)]

    ax.scatter(optotag_df['waveform_duration'], optotag_df['evoked_ratio'],
               s=marker_sizes, marker='o', edgecolors=colors, facecolors='none')
    if evoked_ratio_threshold is not None:
        ax.axhline(evoked_ratio_threshold, linestyle=':', color='r', label='evoked ratio threshold')
    if spike_width_range[0] > 0:
        ax.axvline(spike_width_range[0], linestyle=':', color='g', label='spike width lower bound')
    if spike_width_range[1] < np.inf:
        ax.axvline(spike_width_range[1], linestyle=':', color='orange', label='spike width upper bound')
    ax.scatter([], [], s=alpha_size, marker='o', edgecolors='r', facecolors='none',
        label=f'positive units (n={positive_units.sum():d})')
    ax.scatter([], [], s=alpha_size, marker='o', edgecolors='b', facecolors='none',
        label=f't-test alpha level: {ttest_alpha:.2f}\nmarker size indicates p-value')
    ax.set_yscale('log')
    ax.set_xlabel('Waveform duration (ms)')
    ax.set_ylabel('Evoked ratio')
    ax.legend(loc='upper right', framealpha=0.2)
    return ax
