import numpy as np
import pandas as pd
import xarray as xr

"""
Functions for preprocessing data obtained from allensdk
"""

def align_trials(lfp_array, presentation_ids, onset_times, window=(0., 1.)):
    """Extract and align LFP to time window relative to stimulus onset in given presentations"""
    trial_window = np.arange(window[0], window[1], 1 / lfp_array.fs)
    time_selection = np.concatenate([trial_window + t for t in onset_times])
    inds = pd.MultiIndex.from_product((presentation_ids, trial_window), 
                                      names=('presentation_id', 'time_from_presentation_onset'))
    aligned_lfp = lfp_array.sel(time=time_selection, method='nearest')
    aligned_lfp = aligned_lfp.assign(time=inds).unstack('time')
    return aligned_lfp

def align_gratings(stimulus_presentations, stimulus_name='drifting_gratings'):
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

def align_scenes(stimulus_presentations, stimulus_name='natural_scenes'):
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

def align_movie(stimulus_presentations, stimulus_name='natural_movie_one'):
    """Extract presentations in natural movies stimulus type"""
    presentations = stimulus_presentations[stimulus_presentations.stimulus_name == stimulus_name]
    frame_ids = presentations['stimulus_condition_id'].unique()

    presentations_times = natural_movie_times = np.column_stack([
        presentations[presentations['stimulus_condition_id'] == frame_ids[0]]['start_time'].values,
        presentations[presentations['stimulus_condition_id'] == frame_ids[-1]]['stop_time'].values,
    ])
    presentations_ids = presentations[presentations['stimulus_condition_id'] == frame_ids[0]].index.values
    trial_duration = np.diff(presentations_times, axis=1).max()
    return presentations, presentations_ids, presentations_times[:, 0], trial_duration

def presentation_conditions(presentations, condtion_types):
    """Separate conditions in given presentations and return maps of conditions
    condition_id: map from condition types to condition id
    cond_presentation_id: map from condition to list of presention ids
    """
    conditions = {c: np.unique(presentations[c]).astype(float) for c in condtion_types}
    cond_id_map = dict(zip(map(tuple, presentations[conditions.keys()].values),
                           presentations['stimulus_condition_id']))
    condition_id = [cond_id_map[x] for x in zip(*map(np.ravel, np.meshgrid(*conditions.values(), indexing='ij')))]
    condition_id = xr.DataArray(np.reshape(condition_id, tuple(map(len, conditions.values()))), coords=conditions, name='condition_id')
    cond_presentation_id = {c: presentations.index[presentations['stimulus_condition_id'] == c] for c in condition_id.values.ravel()}
    return condition_id, cond_presentation_id

def get_units_firing_rate(session, stimulus_presentation_id, unit_ids, condition_id, cond_presentation_id,
                          bin_width=0.03, window=(-1.0, 1.0)):
    """Get unit spike time histograms and convert to firing rates"""
    bin_edges = np.concatenate((np.arange(-bin_width / 2, window[0] - bin_width / 2, -bin_width)[::-1],
                            np.arange(bin_width / 2, window[1] + bin_width / 2, bin_width)))
    # get spike count histogram for each unit
    units_fr = session.presentationwise_spike_counts(stimulus_presentation_ids=stimulus_presentation_id,
                                                     unit_ids=unit_ids, bin_edges=bin_edges)
    # average over trials
    units_fr = [units_fr.sel(stimulus_presentation_id=i).mean(dim='stimulus_presentation_id') for i in cond_presentation_id.values()]
    # collect different conditions
    units_fr = xr.concat(units_fr, dim=pd.Index(cond_presentation_id, name='condition_id'))
    # convert to firing rate in Hz
    units_fr = (units_fr / bin_width).to_dataset(name='spike_rate').assign_attrs(bin_width=bin_width)
    # units mean firing rate
    units_fr = units_fr.assign(units_mean_fr=units_fr.spike_rate.mean(dim=['condition_id', 'time_relative_to_stimulus_onset']))
    return units_fr
