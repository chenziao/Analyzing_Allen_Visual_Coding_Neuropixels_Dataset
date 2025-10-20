
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from .format import UNIT

from typing import Literal
from matplotlib.pyplot import Figure, Axes
from numpy.typing import ArrayLike


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


def _get_psd_plt_range(
    plt_range : ArrayLike | float,
    frequencies : ArrayLike | None = None,
    log_frequency : bool = False
) -> tuple[float, float]:
    """Get frequency range for plotting PSD.
    
    Parameters
    ----------
    plt_range : ArrayLike | float
        If float, use (0, plt_range) Hz.
        If ArrayLike, use (plt_range[0], plt_range[1]) Hz.
        if ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    frequencies : ArrayLike | None
        Frequency array for getting frequency range.
    log_frequency : bool
        Whether using log frequency scale (remove 0 Hz from range if True).

    Returns
    -------
    tuple[float, float]
        Frequency range for plotting.
    """
    if frequencies is None:
        freq_range = (0.1 if log_frequency else 0., np.inf)
    else:
        frequencies = np.asarray(frequencies)
        if log_frequency and frequencies[0] == 0:
            freq_range = (frequencies[1], frequencies[-1])
        else:
            freq_range = (frequencies[0], frequencies[-1])
    plt_range = np.asarray(plt_range, dtype=float)
    if plt_range.size == 0:
        plt_range = (0., np.inf)  # full frequency range
    elif plt_range.size == 1:
        plt_range = (0, plt_range.item())
    plt_range = (max(plt_range[0], freq_range[0]), min(plt_range[1], freq_range[1]))
    return plt_range


def plot_channel_psd(
    psd_da : xr.DataArray,
    channel_dim : str = 'channel',
    plt_range : ArrayLike | float = 100.,
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
    plt_range : ArrayLike | float
        Frequency range to plot. Tuple for frequency range or single value for upper limit.
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
    plt_range = _get_psd_plt_range(plt_range, frequencies=psd_da.frequency, log_frequency=log_frequency)
    psd_plt_da = psd_da.sel(frequency=slice(*plt_range))

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
    ax.set_xlim(plt_range)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(f'PSD ({unit})')
    ax.legend(loc='upper right', framealpha=0.2, title=channel_title)
    return ax

