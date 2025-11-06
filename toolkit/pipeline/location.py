import numpy as np
import pandas as pd
import xarray as xr

import warnings

from typing import Sequence


def argsort_channels(
    channels : Sequence[int],
    channels_df : pd.DataFrame,
    sort_by : str = 'probe_vertical_position',
    descending : bool = True
) -> np.ndarray:
    """Argsort channels by a column in `channels_df`.
    
    Parameters
    ----------
    channels : Sequence[int]
        IDs of channels to sort.
    channels_df : pd.DataFrame
        DataFrame of channels with column `sort_by` to sort by.
    sort_by : str
        Column name in `session.channels` to sort the central channels by.
        Default is to sort by vertical position in descending order (superficial to deep).
    descending : bool
        Whether to sort the channels in descending order.

    Returns
    -------
    np.ndarray
        Indices of sorted channels.
    """
    channel_values = channels_df.loc[list(channels), sort_by]
    if any(channel_values.isna()):
        raise ValueError("Channels have missing values in `sort_by` column")
    idx = channel_values.argsort()
    if descending:
        idx = idx[::-1]
    return idx


def get_layer_channel_groups(
    channel_ids : pd.Index,
    central_channels : dict,
    width : int = 0
) -> xr.DataArray:
    """Get groups of channels around central channel of each layer
    
    Parameters
    ----------
    channel_ids : ArrayLike
        IDs of linear channels ordered by position.
        Must include channels within distance `width` of each central channel.
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
