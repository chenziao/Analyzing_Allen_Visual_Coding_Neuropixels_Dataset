from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from ..analysis.signal import bandpass_filter

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..allen_helpers.stimuli import StimulusBlock


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
