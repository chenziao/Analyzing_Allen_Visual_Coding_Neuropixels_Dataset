"""
Functions for preprocessing data obtained from allensdk
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
from collections import namedtuple

from typing import Sequence
from numpy.typing import ArrayLike, NDArray


# Align trials for different stimulus types

def align_flashes(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'flashes'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in flashes stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].median()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_gratings(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'drifting_gratings'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in grating stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    null_rows = presentations[presentations['orientation'].values == 'null']
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].median()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_scenes(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_scenes'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in natural scenes stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    null_rows = presentations[presentations['frame'].values < 0]
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    presentations_times = presentations['start_time'].values
    presentations_ids = presentations.index.values
    trial_duration = presentations['duration'].median()
    return presentations, presentations_ids, presentations_times, trial_duration


def align_movie(
    stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_movie_one'
) -> tuple[pd.DataFrame, NDArray[int], NDArray[float], float]:
    """Extract presentations in natural movies stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    frame_ids = presentations['stimulus_condition_id'].unique()

    presentations_times = np.column_stack([
        presentations[presentations['stimulus_condition_id'] == frame_ids[0]]['start_time'].values,
        presentations[presentations['stimulus_condition_id'] == frame_ids[-1]]['stop_time'].values,
    ])
    presentations_ids = presentations[presentations['stimulus_condition_id'] == frame_ids[0]].index.values
    trial_duration = np.diff(presentations_times, axis=1).median()
    return presentations, presentations_ids, presentations_times[:, 0], trial_duration


def initial_spontaneous_window(
    stimulus_presentations : pd.DataFrame, duration : float | None = 240.
) -> tuple[float, float]:
    """Get the window of the initial spontaneous period (before flashes)
    
    Parameters
    ----------
    stimulus_presentations : pd.DataFrame
        Stimulus presentations.
    duration : float | None
        Duration of the initial spontaneous period. If None, the entire spontaneous period is returned.
    
    Returns
    -------
    window : tuple[float, float]
        Window of the initial spontaneous period. The window is centered around the spontaneous block before flashes.
    """
    flashes_presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == 'flashes']
    flashes_start_time = flashes_presentations.iloc[0]['start_time']
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == 'spontaneous']
    # the spontaneous block before flashes
    initial_spontaneous = presentations[presentations['start_time'] < flashes_start_time].iloc[-1]
    window = initial_spontaneous['start_time'], initial_spontaneous['stop_time']
    if duration is not None:
        excluded_duration = (window[1] - window[0] - duration) / 2
        if excluded_duration < 0:
            raise ValueError(f"Spontaneous duration is shorter than {duration:g} seconds")
        window = (window[0] + excluded_duration, window[1] - excluded_duration)
    return window


ALIGN_FUNCTIONS = {
    'flashes': align_flashes,
}


# General processing functions

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
    aligned_signal.attrs.update(signal_array.attrs)
    return aligned_signal


def align_trials_from_blocks(
    signal_array : xr.DataArray | xr.Dataset,
    stimulus_blocks : list[StimulusBlock],
    window : tuple[float, float] = (0., 1.)
) -> xr.DataArray:
    """Extract and align signal to time window relative to stimulus onset in given blocks"""
    aligned_signals = []
    for stimulus_block in stimulus_blocks:
        aligned_signals.append(align_trials(signal_array, stimulus_block.ids, stimulus_block.times, window))
    aligned_signal = xr.concat(aligned_signals, dim='presentation_id', combine_attrs='override')
    return aligned_signal


StimulusBlock = namedtuple(
    'StimulusBlock',
    ['presentations', 'block_id', 'block_window', 'ids', 'times', 'trial_duration']
)

def get_stimulus_blocks(
    presentations : pd.DataFrame,
    presentations_ids : NDArray[int],
    presentations_times : NDArray[float],
    trial_duration : float
) -> list[StimulusBlock]:
    """Get stimulus blocks from presentations. Split presentations ids and times into blocks.
    
    Parameters
    ----------
    presentations, presentations_ids, presentations_times, trial_duration :
        Output of align functions for different stimulus types.

    Returns
    -------
    stimulus_blocks : list[StimulusBlock]
        Stimulus blocks.
    """
    block_ids = presentations.loc[presentations_ids, 'stimulus_block'].unique()
    stimulus_blocks = []
    for block_id in block_ids:
        block_presentations = presentations.loc[presentations['stimulus_block'] == block_id]
        block_window = (block_presentations['start_time'].iloc[0], block_presentations['stop_time'].iloc[-1])

        block_idx = np.nonzero(presentations.loc[presentations_ids, 'stimulus_block'].values == block_id)[0]
        stimulus_block = StimulusBlock(
            presentations=block_presentations,
            block_id=block_id,
            block_window=block_window,
            ids=presentations_ids[block_idx],
            times=presentations_times[block_idx],
            trial_duration=trial_duration,
        )
        stimulus_blocks.append(stimulus_block)
    return stimulus_blocks
