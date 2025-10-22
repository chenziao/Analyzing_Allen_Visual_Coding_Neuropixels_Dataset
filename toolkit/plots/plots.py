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
    ccf_coordinates : ArrayLike | None = None,
    ccf_label : str = 'Dorsal-Ventral',
    ccf_tick_step : int = 4,
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
    ccf_coordinates : ArrayLike | None
        CCF coordinates of the channels. If specified, show the CCF coordinates on y-axis.
    ccf_label : str
        Label for the CCF axis. Default is 'Dorsal-Ventral'.
    ccf_tick_step : int
        Step for the CCF tick labels (number of channels between ticks).
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
    if ccf_coordinates is not None:
        if central_channels is not None:
            ax.spines['left'].set_position(('outward', 50))  # Move the labels outward
            ax2 = ax.twinx()  # Add a second y-axis with CCF labels
            ax2.set_ylim(ax.get_ylim())
            ax2.yaxis.set_ticks_position('left')
            ax2.yaxis.set_label_position('left')
        else:
            ax2 = ax

        ax2.set_ylabel(f'{ccf_label} CCF ({UNIT.um})', labelpad=0)

        # Choose a subset for labeling every 4th channel
        subset_idx = np.arange(0, len(channel_positions), ccf_tick_step)
        subset_labels = map(str, np.round(np.asarray(ccf_coordinates)[subset_idx]).astype(int))
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

    ax.plot(psd_plt_da.frequency, psd_plt_da.values.T, label=channels)

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

