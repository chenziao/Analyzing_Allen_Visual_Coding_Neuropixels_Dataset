import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp

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

    presentations_times = np.column_stack([
        presentations[presentations['stimulus_condition_id'] == frame_ids[0]]['start_time'].values,
        presentations[presentations['stimulus_condition_id'] == frame_ids[-1]]['stop_time'].values,
    ])
    presentations_ids = presentations[presentations['stimulus_condition_id'] == frame_ids[0]].index.values
    trial_duration = np.diff(presentations_times, axis=1).mean()
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

def get_units_spike_counts(session, stimulus_presentation_id, unit_ids, bin_width=0.03, window=(-1.0, 1.0)):
    """Get unit spike counts in time histograms"""
    bin_edges = np.concatenate((np.arange(-bin_width / 2, window[0] - bin_width / 2, -bin_width)[::-1],
                            np.arange(bin_width / 2, window[1] + bin_width / 2, bin_width)))
    
    units_spk_counts = session.presentationwise_spike_counts(
        stimulus_presentation_ids=stimulus_presentation_id, unit_ids=unit_ids, bin_edges=bin_edges)
    return units_spk_counts

def  get_units_firing_rate(session, stimulus_presentation_id, unit_ids, cond_presentation_id,
                          bin_width=0.03, window=(-1.0, 1.0)):
    """Get unit spike time histograms and convert to firing rates"""
    # get spike count histogram for each unit
    units_spk_counts = get_units_spike_counts(session, stimulus_presentation_id, unit_ids,
                                              bin_width=bin_width, window=window)
    # average over trials
    units_fr = [units_spk_counts.sel(stimulus_presentation_id=i).mean(dim='stimulus_presentation_id') \
        for i in cond_presentation_id.values()]
    # collect different conditions
    units_fr = xr.concat(units_fr, dim=pd.Index(cond_presentation_id, name='condition_id'))
    # convert to firing rate in Hz
    spike_rate = units_fr / bin_width
    units_fr_min = spike_rate.min(dim=('condition_id', 'time_relative_to_stimulus_onset'))
    units_fr_max = spike_rate.max(dim=('condition_id', 'time_relative_to_stimulus_onset'))
    units_fr = spike_rate.to_dataset(name='spike_rate').assign_attrs(bin_width=bin_width)
    # units firing rate statistics
    units_fr = units_fr.assign(
        units_fr_mean=spike_rate.mean(dim=['condition_id', 'time_relative_to_stimulus_onset']),
        units_fr_std=spike_rate.std(dim=('condition_id', 'time_relative_to_stimulus_onset')),
        units_fr_min=units_fr_min,
        units_fr_max=units_fr_max,
        units_fr_range = units_fr_max - units_fr_min
    )
    return units_fr

def preprocess_firing_rate(units_fr, sigma, units_fr_mean=None, soft_normalize_cut=0., normalization_scale='range'):
    """Smooth and normalize units firing rate
    units_fr: xarray of unit firing rate data and statistics
    sigma: sigma of gaussian filter
    units_fr_mean: array of mean firing rate of each unit
    soft_normalize_cut: cutoff value of soft-normalization. Set to 0 is equivalent to regular normalization.
    normalization_scale:
        If is a string, use 'range' (default) min-max normalization, 'std' standardization, 'mean' normalize by mean
        Otherwise, should be an array of scaling factor for firing rate normalization of each unit
    """
    if units_fr_mean is None:
        units_fr_mean = units_fr.units_fr_mean
    else:
        units_fr_mean = xr.DataArray(units_fr_mean, coords={'unit_id': units_fr.unit_id})
    if isinstance(normalization_scale, str):
        normalization_scale = getattr(units_fr, 'units_fr_' + normalization_scale)
    else:
        normalization_scale = xr.DataArray(normalization_scale, coords={'unit_id': units_fr.unit_id})
    # smooth firing rate using Gaussian filter
    axis = units_fr.spike_rate.dims.index('time_relative_to_stimulus_onset')
    smoothed = sp.ndimage.gaussian_filter1d(units_fr.spike_rate - units_fr_mean,
                                            sigma / units_fr.bin_width, axis=axis, mode='constant')
    smoothed = units_fr.spike_rate.copy(data=smoothed) + units_fr_mean
    units_fr = units_fr.assign(smoothed=smoothed)
    # soft normalize firing rate
    normalized = smoothed / (normalization_scale + soft_normalize_cut)
    units_fr = units_fr.assign(normalized=normalized)
    return units_fr

def stack_time_samples(da, sample_dims=None, non_sample_dims=None, keep_multi_index=False):
    """Stack multiple dims of time points in a dataarray to a single sample axis along the first dimension
    sample_dims: dimensions along which to consider as samples. default is for spike counts data from allensdk
        Dimensions not in the dataarray will be ignored.
        Note: the order of dimensions in `sample_dims` determines the order of samples
    non_sample_dims: dimensions along which to be excluded as samples.
        If specified, `sample_dims` will be ignored.
        Note: the order of dimensions in the dataarray determines the order of samples
    """
    if non_sample_dims is None:
        if sample_dims is None:
            sample_dims = ('stimulus_presentation_id', 'time_relative_to_stimulus_onset')
        sample_dims = [d for d in sample_dims if d in da.dims]
    else:
        sample_dims = [d for d in da.dims if d not in non_sample_dims]
    da = da.stack(sample=sample_dims)
    dims = list(da.dims)
    dims.remove('sample')
    da = da.transpose('sample', *dims)
    if not keep_multi_index:
        samples = np.arange(da.sample.size)
        da = da.drop_vars(sample_dims)
        da = da.assign_coords(sample=samples)
    return da

def stimuli_data_to_samples(datasets, sample_dims=None, non_sample_dims=None, var='spike_rate'):
    """Concatenate time points from multiple Datasets to a single sample axis
    sample_dims: dimensions along which to consider as samples. default is for spike counts data from allensdk
    average_trials: whether the datasets are trial averaged. Dimension 'stimulus_presentation_id'
        will not be included as sample dimensions if `average_trials` is True.
    var: variable name of DataArray in Dataset to concatenate.
        if is None, assume `datasets` is list of DataArrays
    """
    if var is None or isinstance(next(iter(datasets)), xr.DataArray):
        dataarrays = datasets
    else:
        dataarrays = [ds[var] for ds in datasets]
    da = xr.concat([stack_time_samples(da, sample_dims=sample_dims,
        non_sample_dims=non_sample_dims) for da in dataarrays], dim='sample')
    da = da.assign_coords(sample=range(da.sample.size))
    return da
