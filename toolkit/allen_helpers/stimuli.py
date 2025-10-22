"""
Functions for preprocessing data obtained from allensdk
"""

import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from numpy.typing import NDArray


# Lookup table for stimuli
STIMULUS_NAMES = {
    "brain_observatory_1.1": [
        'flashes',
        'drifting_gratings',
        'natural_movie_three',
        'natural_movie_one',
        'static_gratings',
        'natural_scenes',
    ],
    "functional_connectivity": [
        'flashes',
        'drifting_gratings_contrast',
        'drifting_gratings_75_repeats',
        'natural_movie_one_more_repeats',
        'natural_movie_one_shuffled',
    ]
}

STIMULUS_CATEGORIES = {
    "brain_observatory_1.1": {
        'drifting_gratings': ['drifting_gratings'],
        'natural_movies': ['natural_movie_one', 'natural_movie_three'],
    },
    "functional_connectivity": {
        'drifting_gratings': ['drifting_gratings_75_repeats', 'drifting_gratings_contrast'],
        'natural_movies': ['natural_movie_one_more_repeats', 'natural_movie_one_shuffled']
    }
}


# Drifting gratings conditions
CONDITION_TYPES = ('orientation', 'temporal_frequency', 'contrast')

# condition types with varied/fixed values in each session type
VARIED_CONDITION_TYPES = {
    "brain_observatory_1.1": ('orientation', 'temporal_frequency'),
    "functional_connectivity": ('contrast', 'orientation'),
}
FIXED_CONDITION_TYPES = {
    "brain_observatory_1.1": ('contrast', ),
    "functional_connectivity": ('temporal_frequency', ),
}

# Labels of condition types for displaying
COND_LABEL = dict(
    orientation = 'Orientation (deg)',
    temporal_frequency = 'Temporal frequency (Hz)',
    contrast = 'Contrast'
)


# Get trials for different stimulus types

@dataclass
class StimulusTrials:
    """Stimulus trials containing trial information

    Attributes
    ----------
    presentations: pd.DataFrame
        Stimulus presentations dataframe.
    ids: NDArray[int]
        presentations IDs for each trial.
    times: NDArray[float]
        Onset times for each trial.
    duration: float
        Median duration for each trial.
    """
    presentations: pd.DataFrame
    ids: NDArray[int]
    times: NDArray[float]
    duration: float


def get_flashes_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'flashes') -> StimulusTrials:
    """Extract presentations in flashes stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations.index.values,
        times = presentations['start_time'].values,
        duration = presentations['duration'].median()
    )
    return stimulus_trials


def get_gratings_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'drifting_gratings') -> StimulusTrials:
    """Extract presentations in grating stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    null_rows = presentations[presentations['orientation'].values == 'null']
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations.index.values,
        times = presentations['start_time'].values,
        duration = presentations['duration'].median()
    )
    return stimulus_trials


def get_scenes_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_scenes') -> StimulusTrials:
    """Extract presentations in natural scenes stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    null_rows = presentations[presentations['frame'].values < 0]
    if len(null_rows):
        null_condition = null_rows.iloc[0]['stimulus_condition_id']
        presentations = presentations[presentations['stimulus_condition_id'] != null_condition]

    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations.index.values,
        times = presentations['start_time'].values,
        duration = presentations['duration'].median()
    )
    return stimulus_trials


def get_movie_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_movie_one') -> StimulusTrials:
    """Extract presentations in natural movies stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    frame_ids = presentations['stimulus_condition_id'].unique()

    presentations_times = np.column_stack([
        presentations[presentations['stimulus_condition_id'] == frame_ids[0]]['start_time'].values,
        presentations[presentations['stimulus_condition_id'] == frame_ids[-1]]['stop_time'].values,
    ])
    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations[presentations['stimulus_condition_id'] == frame_ids[0]].index.values,
        times = presentations_times[:, 0],
        duration = np.median(np.diff(presentations_times, axis=1))
    )
    return stimulus_trials


# map stimulus name to function to get stimulus trials

ALIGN_FUNCTIONS = dict(
    # Common stimuli
    flashes = get_flashes_trials,
    # Brain Observatory 1.1
    drifting_gratings = get_gratings_trials,
    natural_movie_three = get_movie_trials,
    natural_movie_one = get_movie_trials,
    static_gratings = get_gratings_trials,
    natural_scenes = get_scenes_trials,
    # Functional Connectivity
    drifting_gratings_contrast = get_gratings_trials,
    drifting_gratings_75_repeats = get_gratings_trials,
    natural_movie_one_more_repeats = get_movie_trials,
    natural_movie_one_shuffled = get_movie_trials
)

def get_stimulus_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str) -> StimulusTrials:
    """Get trials for a given stimulus type"""
    get_trials = ALIGN_FUNCTIONS.get(stimulus_name)
    if get_trials is None:
        raise ValueError(f"Stimulus type '{stimulus_name}' not supported.")
    return get_trials(stimulus_presentations, stimulus_name=stimulus_name)


# Get stimulus blocks for different stimulus types

@dataclass
class StimulusBlock(StimulusTrials):
    """Stimulus block containing trial information and additional block-specific attributes
    
    Attributes
    ----------
    block_id: int
        Block ID.
    block_window: tuple[float, float]
        Block time window (start, end).
    """
    block_id: int
    block_window: tuple[float, float]


def get_stimulus_blocks(stimulus_trials : StimulusTrials) -> list[StimulusBlock]:
    """Get stimulus blocks from presentations. Split presentations ids and times into blocks.
    
    Parameters
    ----------
    stimulus_trials : StimulusTrials
        Output of `get_stimulus_trials()` for a given stimulus type.

    Returns
    -------
    stimulus_blocks : list[StimulusBlock]
        Stimulus blocks.
    """
    presentations = stimulus_trials.presentations
    block_ids = presentations.loc[stimulus_trials.ids, 'stimulus_block'].unique()
    stimulus_blocks = []
    for block_id in block_ids:
        block_presentations = presentations.loc[presentations['stimulus_block'] == block_id]
        block_window = (block_presentations['start_time'].iloc[0], block_presentations['stop_time'].iloc[-1])

        block_idx = np.nonzero(presentations.loc[stimulus_trials.ids, 'stimulus_block'].values == block_id)[0]
        stimulus_block = StimulusBlock(
            presentations = block_presentations,
            ids = stimulus_trials.ids[block_idx],
            times = stimulus_trials.times[block_idx],
            duration = stimulus_trials.duration,
            block_id = block_id,
            block_window = block_window,
        )
        stimulus_blocks.append(stimulus_block)
    return stimulus_blocks


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


# General processing functions

def align_trials(
    signal_array : xr.DataArray | xr.Dataset,
    stimulus_trials : StimulusTrials,
    window : tuple[float, float] = (0., 1.)
) -> xr.DataArray | xr.Dataset:
    """Extract and align signal to time window relative to stimulus onset in given presentations
    
    Parameters
    ----------
    signal_array : xr.DataArray | xr.Dataset
        Signal array. Attributes must include 'fs'.
    stimulus_trials : StimulusTrials
        Stimulus trials.
    window : tuple[float, float]
        Window relative to stimulus onset to extract and align signal to.

    Returns
    -------
    aligned_signal : xr.DataArray | xr.Dataset
        Aligned signal. Dimension: presentation_id, time_from_presentation_onset.
    """
    fs = signal_array.attrs['fs']
    window_idx = int(np.floor(window[0] * fs)), int(np.ceil(window[1] * fs))
    trial_window = np.arange(window_idx[0], window_idx[1])
    onset_indices = signal_array.get_index('time').get_indexer(stimulus_trials.times, method='nearest')
    time_selection = np.concatenate([trial_window + t for t in onset_indices])
    aligned_signal = signal_array.isel(time=time_selection)
    inds = pd.MultiIndex.from_product(
        (stimulus_trials.ids, trial_window / fs),
        names=('presentation_id', 'time_from_presentation_onset')
    )
    aligned_signal = aligned_signal.assign_coords(time=inds).unstack('time')
    aligned_signal.attrs.update(signal_array.attrs)
    return aligned_signal


def align_trials_from_blocks(
    signal_array : xr.DataArray | xr.Dataset,
    stimulus_blocks : list[StimulusBlock],
    window : tuple[float, float] = (0., 1.)
) -> xr.DataArray | xr.Dataset:
    """Extract and align signal to time window relative to stimulus onset in given blocks.
    Similar to `align_trials()`, but for multiple blocks.
    """
    aligned_signals = []
    for stimulus_block in stimulus_blocks:
        aligned_signals.append(align_trials(signal_array, stimulus_block, window))
    aligned_signal = xr.concat(aligned_signals, dim='presentation_id', combine_attrs='override')
    return aligned_signal


def presentation_conditions(
    presentations : pd.DataFrame,
    condition_types : tuple[str, ...]
) -> tuple[xr.DataArray, dict[str, NDArray[int]]]:
    """Separate conditions in given presentations and return maps of conditions

    Parameters
    ----------
    presentations : pd.DataFrame
        Stimulus presentations.
    condition_types : tuple[str, ...]
        Condition types of interest.

    Returns
    -------
    condition_id : xr.DataArray
        Condition id. Dimension: condition types. Coordinates: unique values of each condition type.
    cond_presentation_id : dict[str, NDArray[int]]
        Map from condition to presentation ids of the condition.
    """
    # map condition type to unique values of the condition
    conditions = {c: np.unique(presentations[c]).astype(float) for c in condition_types}
    # map value tuple of different condition types to stimulus condition id
    cond_id_map = dict(zip(
        map(tuple, presentations[conditions.keys()].values),
        presentations['stimulus_condition_id']
    ))
    # construct condition id DataArray
    condition_mesh = np.array(np.meshgrid(*conditions.values(), indexing='ij'))  # (condition_type, mesh grid ...)
    condition_combinations = condition_mesh.reshape(condition_mesh.shape[0], -1).T  # (all conditions, values combination)
    condition_id = [cond_id_map[tuple(x)] for x in condition_combinations]
    condition_id = np.reshape(condition_id, condition_mesh.shape[1:])  # condition_id in mesh grid shape
    condition_id = xr.DataArray(condition_id, coords=conditions, name='condition_id')  # condition values as coordinates
    # map condition id to presentation ids of the condition
    presentations_id = presentations.index.values
    stimulus_condition_id = presentations['stimulus_condition_id'].values
    cond_presentation_id = {c: presentations_id[stimulus_condition_id == c] for c in np.ravel(condition_id)}
    return condition_id, cond_presentation_id


def average_across_conditions(
    da : xr.DataArray | xr.Dataset,
    condition_id : xr.DataArray,
    cond_presentation_id : dict[str, NDArray[int]]
) -> xr.DataArray | xr.Dataset:
    """Average data across conditions.
    
    Parameters
    ----------
    da : xr.DataArray | xr.Dataset
        Data array to be averaged across conditions. Must include 'presentation_id' dimension.
    condition_id : xr.DataArray
        Condition id. Returned by `presentation_conditions()`.
    cond_presentation_id : dict[str, NDArray[int]]
        Map from condition to presentation ids of the condition. Returned by `presentation_conditions()`.

    Returns
    -------
    cond_da : xr.DataArray | xr.Dataset
        Data array averaged across conditions. Dimension 'presentation_id' is removed.
        New dimension is 'condition_id', which itself has dimensions of the condition types,
        with coordinates being the condition values.
    """
    cond_da = [da.sel(presentation_id=i).mean(dim='presentation_id') for i in cond_presentation_id.values()]
    cond_da = xr.concat(cond_da, dim=pd.Index(cond_presentation_id, name='condition_id'), combine_attrs='override')
    cond_da = cond_da.sel(condition_id=condition_id)
    return cond_da

