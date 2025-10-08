
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from .format import UNIT

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
