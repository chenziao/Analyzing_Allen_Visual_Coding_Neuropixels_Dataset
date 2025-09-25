"""
Functions for preprocessing data obtained from allensdk
"""

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp

from typing import Sequence
from numpy.typing import ArrayLike, NDArray


def align_trials(
    signal_array : xr.DataArray | xr.Dataset,
    presentation_ids : Sequence[int],
    onset_times : Sequence[float],
    window : tuple[float, float] = (0., 1.)
) -> xr.DataArray:
    """Extract and align signal to time window relative to stimulus onset in given presentations"""
    fs = signal_array.fs
    window_idx = int(np.floor(window[0] *fs)), int(np.ceil(window[1] * fs))
    trial_window = np.arange(window_idx[0], window_idx[1])
    onset_indices = signal_array.get_index('time').get_indexer(onset_times, method='nearest')
    time_selection = np.concatenate([trial_window + t for t in onset_indices])
    aligned_signal = signal_array.isel(time=time_selection)
    inds = pd.MultiIndex.from_product((presentation_ids, trial_window / signal_array.fs),
                                       names=('presentation_id', 'time_from_presentation_onset'))
    aligned_signal = aligned_signal.assign_coords(time=inds).unstack('time')
    return aligned_signal


def align_flashes(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'flashes'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in flashes stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations.stimulus_name == stimulus_name]
    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].max()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_gratings(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'drifting_gratings'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in grating stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations.stimulus_name == stimulus_name]
    null_rows = presentations[presentations['orientation'].values == 'null']
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].max()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_scenes(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_scenes'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in natural scenes stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations.stimulus_name == stimulus_name]
    null_rows = presentations[presentations['frame'].values < 0]
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].max()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_movie(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_movie_one'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in natural movies stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations.stimulus_name == stimulus_name]
    frame_ids = presentations['stimulus_condition_id'].unique()

    presentations_times = np.column_stack([
        presentations[presentations['stimulus_condition_id'] == frame_ids[0]]['start_time'].values,
        presentations[presentations['stimulus_condition_id'] == frame_ids[-1]]['stop_time'].values,
    ])
    presentations_ids = presentations[presentations['stimulus_condition_id'] == frame_ids[0]].index.values
    trial_duration = np.diff(presentations_times, axis=1).mean()
    return presentations, presentations_ids, presentations_times[:, 0], trial_duration

