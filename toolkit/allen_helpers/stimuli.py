"""
Functions for preprocessing data obtained from allensdk
"""

import numpy as np
import pandas as pd
import xarray as xr

from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike


# Lookup table for stimuli
STIMULUS_NAMES = {
    "brain_observatory_1.1": [
        'spontaneous',
        'flashes',
        'drifting_gratings',
        'natural_movie_three',
        'natural_movie_one',
        'static_gratings',
        'natural_scenes',
    ],
    "functional_connectivity": [
        'spontaneous',
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

STIMULUS_SESSION_TYPES = {}
for session_type, category in STIMULUS_CATEGORIES.items():
    for c in category.values():
        for stim in c:
            STIMULUS_SESSION_TYPES[stim] = session_type


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
    gap_duration: float
        Minimum gap duration between trials.
    presentation_increment: int
        Increment between consecutive presentations.
    """
    presentations: pd.DataFrame
    ids: NDArray[int]
    times: NDArray[float]
    duration: float
    gap_duration: float
    presentation_increment: int = 1


def get_flashes_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'flashes') -> StimulusTrials:
    """Extract presentations in flashes stimulus types"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations.index.values,
        times = presentations['start_time'].values,
        duration = presentations['duration'].median(),
        gap_duration = np.min(presentations['start_time'].values[1:] - presentations['stop_time'].values[:-1])
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
        duration = presentations['duration'].median(),
        gap_duration = np.min(presentations['start_time'].values[1:] - presentations['stop_time'].values[:-1])
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
        duration = presentations['duration'].median(),
        gap_duration = np.min(presentations['start_time'].values[1:] - presentations['stop_time'].values[:-1])
    )
    return stimulus_trials


def get_movie_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'natural_movie_one') -> StimulusTrials:
    """Extract presentations in natural movies stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    frame_ids = presentations['stimulus_condition_id'].unique()
    presentation_increment = frame_ids.size

    start_idx = np.nonzero(presentations['stimulus_condition_id'].values == frame_ids[0])[0]
    stop_idx = np.nonzero(presentations['stimulus_condition_id'].values == frame_ids[-1])[0]
    start_times = presentations['start_time'].values[start_idx]
    stop_times = presentations['stop_time'].values[stop_idx]
    # Pair monotonic start/stop times where stop > start
    presentations_times = []
    presentation_idx = []
    i, j = 0, 0
    while i < start_times.size and j < stop_times.size:
        if stop_times[j] > start_times[i]:
            if i + 1 >= start_times.size or start_times[i + 1] >= stop_times[j]:
                # this start time is the last start time for this stop time
                presentations_times.append([start_times[i], stop_times[j]])
                presentation_idx.append([start_idx[i], stop_idx[j] + 1])
                j += 1
            i += 1
        else:
            j += 1
    presentations_times = np.array(presentations_times)
    presentation_idx = np.array(presentation_idx)  # slice indices of each trial
    # check missing presentations by presentation_increment
    idx = np.nonzero(presentation_idx[:, 1] - presentation_idx[:, 0] == presentation_increment)[0]
    presentations_times = presentations_times[idx]
    presentation_idx = presentation_idx[idx]
    stimulus_trials = StimulusTrials(
        presentations = presentations.iloc[np.concatenate([np.arange(i, j) for i, j in presentation_idx])],
        ids = presentations.iloc[presentation_idx[:, 0]].index.values,
        times = presentations_times[:, 0],
        duration = np.median(np.diff(presentations_times, axis=1)),
        gap_duration = np.min(presentations_times[1:, 0] - presentations_times[:-1, 1]),
        presentation_increment = presentation_increment
    )
    return stimulus_trials


def initial_spontaneous_presentation(
    stimulus_presentations : pd.DataFrame,
    duration : float | None = 240.,
    stimulus_name : str = 'spontaneous'
) -> tuple[float, float]:
    """Get the presentation and window of the longest initial spontaneous period (before flashes)
    
    Parameters
    ----------
    stimulus_presentations : pd.DataFrame
        Stimulus presentations.
    duration : float | None
        Duration of the initial spontaneous period. If None, the entire spontaneous period is returned.
    stimulus_name : str
        Stimulus name. Default is 'spontaneous'.
    
    Returns
    -------
    presentations : pd.DataFrame
        Presentations of the initial spontaneous period.
    window : tuple[float, float]
        Window of the initial spontaneous period. The window is centered around the spontaneous block before flashes.
    """
    flashes_presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == 'flashes']
    flashes_start_time = flashes_presentations.iloc[0]['start_time']
    presentations = stimulus_presentations[stimulus_presentations['stimulus_name'] == stimulus_name]
    # the longest spontaneous presentation before flashes
    presentations = presentations[presentations['start_time'] < flashes_start_time]
    presentations = presentations.iloc[[np.argmax(presentations['duration'])]]
    window = presentations.iloc[0]['start_time'], presentations.iloc[0]['stop_time']
    # restrict window to given duration
    if duration is not None:
        excluded_duration = (window[1] - window[0] - duration) / 2
        if excluded_duration < 0:
            raise ValueError(f"Spontaneous duration can be at most {duration:g} seconds")
        window = (window[0] + excluded_duration, window[1] - excluded_duration)
    return presentations, window


def get_spontaneous_trials(stimulus_presentations : pd.DataFrame, stimulus_name : str = 'spontaneous') -> StimulusTrials:
    """Extract presentations (longest initial spontaneous period) in spontaneous stimulus type"""
    presentations, window = initial_spontaneous_presentation(stimulus_presentations, stimulus_name=stimulus_name)
    stimulus_trials = StimulusTrials(
        presentations = presentations,
        ids = presentations.index.values,
        times = np.array([window[0]]),
        duration = window[1] - window[0],
        gap_duration = 0.
    )
    return stimulus_trials

# map stimulus name to function to get stimulus trials

ALIGN_FUNCTIONS = dict(
    # Common stimuli
    spontaneous = get_spontaneous_trials,
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
    block_id: int = 0
    sub_block_id: int = 0
    block_window: tuple[float, float] = (0., 0.)


def get_stimulus_blocks(stimulus_trials : StimulusTrials) -> list[StimulusBlock]:
    """Get stimulus blocks from presentations.
    Split presentation dataframe, presentations ids and times into blocks.
    A block is a contiguous set of presentations with the same stimulus block id.
    If any presentations are missing in the block, the block is split into sub-blocks at the missing presentation(s).

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
    presentation_increment = stimulus_trials.presentation_increment
    block_ids = presentations.loc[stimulus_trials.ids, 'stimulus_block'].unique()
    stimulus_blocks = []
    for block_id in block_ids:
        # all presentations in the block
        block_presentations = presentations.loc[presentations['stimulus_block'] == block_id]
        # representative presentation ids in the block with presentation increment
        block_presentations_idx = presentations.loc[stimulus_trials.ids, 'stimulus_block'].values == block_id
        block_presentations_ids = stimulus_trials.ids[block_presentations_idx]
        block_times = stimulus_trials.times[block_presentations_idx]
        # Split presentation ids at missing presentations (greater than presentation increment)
        split_idx = np.nonzero(np.diff(block_presentations_ids) > presentation_increment)[0] + 1
        interval_idx = np.column_stack((np.insert(split_idx, 0, 0), np.append(split_idx, block_presentations_ids.size)))

        for sub_block_id, (start_idx, end_idx) in enumerate(interval_idx):
            ids = block_presentations_ids[start_idx:end_idx]
            sub_block_presentations = block_presentations.loc[
                (block_presentations.index >= ids[0]) & \
                (block_presentations.index < ids[-1] + presentation_increment)
            ]
            block_window = (sub_block_presentations['start_time'].iloc[0],
                sub_block_presentations['stop_time'].iloc[-1])
            stimulus_block = StimulusBlock(
                presentations = sub_block_presentations,
                ids = ids,
                times = block_times[start_idx:end_idx],
                duration = stimulus_trials.duration,
                gap_duration = stimulus_trials.gap_duration,
                presentation_increment = presentation_increment,
                block_id = block_id,
                sub_block_id = sub_block_id,
                block_window = block_window
            )
            stimulus_blocks.append(stimulus_block)
    return stimulus_blocks


# General processing functions

def align_trials(
    signal_array : xr.DataArray | xr.Dataset,
    stimulus_trials : StimulusTrials | StimulusBlock,
    window : tuple[float, float] = (0., 1.),
    ignore_nan_trials : str = 'auto'
) -> tuple[xr.DataArray | xr.Dataset, StimulusTrials | StimulusBlock | None]:
    """Extract and align signal to time window relative to stimulus onset in given presentations
    
    Parameters
    ----------
    signal_array : xr.DataArray | xr.Dataset
        Signal array. Attributes must include 'fs'.
    stimulus_trials : StimulusTrials | StimulusBlock
        Stimulus trials.
    window : tuple[float, float]
        Window relative to stimulus onset to extract and align signal to.
    ignore_nan_trials : str
        How to handle NaN values in trials.
        'any': ignore trials with any NaN values.
        'all': ignore trials with all NaN values.
        'auto': check 'any' first, and fall back to 'all' if all trials have any NaN values.
        '' or any other value: do not ignore NaN values.

    Returns
    -------
    aligned_signal : xr.DataArray | xr.Dataset
        Aligned signal. Dimension: presentation_id, time_from_presentation_onset.
    valid_trials : StimulusTrials | StimulusBlock | None
        Valid trials. If no trial is dropped, return None.
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

    if ignore_nan_trials not in {'auto', 'any', 'all'}:
        ignore_nan_trials = ''  # not ignore nan trials

    non_trial_dims = [d for d in aligned_signal.dims if d != 'presentation_id']
    if ignore_nan_trials == 'auto' or ignore_nan_trials == 'any':
        # check if each trial has any nan
        nan_trials = np.isnan(aligned_signal).any(dim=non_trial_dims).values
        # fall back to 'all' if 'auto' and all trials have any nan
        if ignore_nan_trials == 'auto' and nan_trials.all():
            ignore_nan_trials = 'all'
            stimulus_name = stimulus_trials.presentations.iloc[0]['stimulus_name']
            print(f"All trials have any NaN values in {stimulus_name}. Fall back to 'all'.")
        else:
            ignore_nan_trials = 'any'
    if ignore_nan_trials == 'all':
        # check if each trial has all nan
        nan_trials = np.isnan(aligned_signal).all(dim=non_trial_dims).values
    if ignore_nan_trials and nan_trials.any():
        valid_trials = choose_trials(stimulus_trials, ~nan_trials)
        aligned_signal = aligned_signal.sel(presentation_id=valid_trials.ids)
        stimulus_name = stimulus_trials.presentations.iloc[0]['stimulus_name']
        print(f"Dropped {nan_trials.size - valid_trials.ids.size} trials with "
            f"{ignore_nan_trials} NaN values in {stimulus_name}.")
    else:
        valid_trials = None
    return aligned_signal, valid_trials


def choose_trials(
    stimulus_trials : StimulusTrials | StimulusBlock,
    trial_ids : NDArray[bool] | NDArray[int] | ArrayLike
) -> StimulusTrials | StimulusBlock:
    """Choose trials with the given indices
    
    Parameters
    ----------
    stimulus_trials : StimulusTrials | StimulusBlock
        Stimulus trials object to choose trials from.
    trial_ids : NDArray[bool] | NDArray[int] | ArrayLike
        Trial presentation ids or boolean indices of trials.

    Returns
    -------
    valid_trials : StimulusTrials | StimulusBlock
        New StimulusTrials or StimulusBlock object with the chosen trials.
    """
    from copy import copy
    trial_ids = np.asarray(trial_ids)
    if trial_ids.dtype == 'bool':  # if boolean indices, get integer indices
        if trial_ids.size != stimulus_trials.ids.size:
            raise ValueError("Boolean indices must have the same size as the number of trials")
        trial_idx = np.nonzero(trial_ids)[0]
    else:  # if not boolean indices, use it as presentation ids
        trial_idx = pd.Index(stimulus_trials.ids).get_indexer(trial_ids)
    valid_trials = copy(stimulus_trials)
    valid_trials.ids = valid_trials.ids[trial_idx]
    valid_trials.times = valid_trials.times[trial_idx]
    presentation_ids = valid_trials.ids[:, None] + np.arange(valid_trials.presentation_increment)
    valid_trials.presentations = valid_trials.presentations.loc[presentation_ids.ravel()].copy()
    return valid_trials


def align_trials_from_blocks(
    signal_array : xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset],
    stimulus_blocks : list[StimulusBlock],
    window : tuple[float, float] = (0., 1.),
    ignore_nan_trials : str = 'auto'
) -> tuple[xr.DataArray | xr.Dataset, list[StimulusBlock | None]]:
    """Extract and align signal to time window relative to stimulus onset in given blocks.
    Similar to `align_trials()`, but for multiple blocks.
    `signal_array` can be a single xr.DataArray or xr.Dataset,
    or a list of xr.DataArray or xr.Dataset corresponding to the input `stimulus_blocks`.
    Return `valid_blocks`. If no block is dropped, it is the same as the input `stimulus_blocks`.
    If any presentation is missing in the input blocks, the returned blocks include sub-blocks split from the original blocks.
    """
    aligned_signals = []
    valid_blocks = []
    if isinstance(signal_array, xr.DataArray | xr.Dataset):
        signal_array = [signal_array] * len(stimulus_blocks)
    for da, stimulus_block in zip(signal_array, stimulus_blocks):
        aligned_signal, valid_trials = align_trials(da, stimulus_block, window, ignore_nan_trials)
        aligned_signals.append(aligned_signal)
        if valid_trials is None:
            valid_blocks.append(stimulus_block)
        else:
            valid_blocks.extend(get_stimulus_blocks(valid_trials))
    aligned_signal = xr.concat(aligned_signals, dim='presentation_id', combine_attrs='override')
    return aligned_signal, valid_blocks


def presentation_conditions(
    presentations : pd.DataFrame,
    condition_types : tuple[str, ...] = CONDITION_TYPES
) -> tuple[xr.DataArray, dict[str, NDArray[int]]]:
    """Separate conditions in given presentations and return maps of conditions

    Parameters
    ----------
    presentations : pd.DataFrame
        Stimulus presentations.
    condition_types : tuple[str, ...]
        Condition types of interest. Default is ('orientation', 'temporal_frequency', 'contrast').

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


def average_trials_with_conditions(
    da : xr.DataArray | xr.Dataset,
    condition_id : xr.DataArray,
    cond_presentation_id : dict[str, NDArray[int]]
) -> xr.DataArray | xr.Dataset:
    """Average data with conditions across presentations.
    
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
    cond_da = [da.sel(presentation_id=i).mean(dim='presentation_id', keep_attrs=True) for i in cond_presentation_id.values()]
    cond_da = xr.concat(cond_da, dim=pd.Index(cond_presentation_id, name='condition_id'), combine_attrs='override')
    return cond_da.sel(condition_id=condition_id)

