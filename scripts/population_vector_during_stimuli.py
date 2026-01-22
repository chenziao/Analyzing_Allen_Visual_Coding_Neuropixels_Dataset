"""
Population vector during stimuli (during drifting gratings and natural movies).

- Compute normalized and downsampled LFP band power during stimuli.
- Get spike rate of units during stimuli.
- Get firing rate statistics of units and compute normalized firing rate.
- Perform PCA and compute population vector.
- Save data.
"""

import add_path

import numpy as np


PARAMETERS = dict(
    config_name_suffix = dict(
        default = '',
        type = str,
        help = "Suffix for the configuration name."
    ),
    overwrite_spike_counts = dict(
        default = False,
        type = bool,
        help = "Whether to overwrite `units_spike_rate` data file."
    ),
    combine_stimulus_name = dict(
        default = 'mixed_stimuli',
        type = str,
        help = "Stimuli combination for population vector analysis. "
            "'natural_movies' or 'drifting_gratings' or 'mixed_stimuli'."
    ),
    filter_orientation = dict(
        default = False,
        type = bool,
        help = "Whether to filter conditions by preferred orientation."
    ),
    sigma = dict(
        default = 0,
        type = float,
        help = "Gaussian filter sigma in seconds for smoothing firing rate." 
            "If 0, set to bin width."
    ),
    normalize_unit_fr = dict(
        default = True,
        type = bool,
        help = "Whether to normalize unit firing rate."
    ),
    soft_normalize = dict(
        default = True,
        type = bool,
        help = "Whether to use soft normalization."
    ),
    normalization_scale = dict(
        default = 'std',
        type = str,
        help = "Normalization scale for unit firing rate. 'max', 'std', 'mean'."
    ),
    quantile = dict(
        default = 0.2,
        type = float,
        help = "Quantile for soft normalization cutoff."
    ),
    select_RS = dict(
        default = False,
        type = bool,
        help = "Whether to select only Regular Spiking (RS) units."
    ),
    select_layer = dict(
        default = False,
        type = bool,
        help = "Whether to select units the layer of interest."
    ),
    absolute_origin = dict(
        default = True,
        type = bool,
        help = "Whether to use absolute zeros firing rate as the origin in principal space."
    ),
    n_pc_range = dict(
        default = [3, 6],
        type = list,
        list_type = int,
        help = "Range of number of principal components to consider."
    )
)



def vec_len(da, dim='PC'):
    """Euclidean distance along a dimension"""
    return np.sqrt((da ** 2).sum(dim=dim))


def vec_dot(da1, da2, dim='PC'):
    """Dot product along a dimension"""
    return (da1 * da2).sum(dim=dim)


def population_vector_during_stimuli(
    session_id: int,
    config_name_suffix: str = '',
    overwrite_spike_counts: bool = False,
    combine_stimulus_name: str = 'mixed_stimuli',
    filter_orientation: bool = False,
    sigma: float = 0,
    normalize_unit_fr: bool = True,
    soft_normalize: bool = True,
    normalization_scale: str = 'std',
    quantile: float = 0.2,
    select_RS: bool = True,
    select_layer: bool = False,
    absolute_origin: bool = True,
    n_pc_range: list[int] = [3, 6],
    is_main_job: bool = True,
) -> None:
    import json
    import pandas as pd
    import xarray as xr

    import toolkit.allen_helpers.stimuli as st
    import toolkit.pipeline.signal as ps
    from toolkit.pipeline.data_io import SessionDirectory, FILES, safe_mkdir
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS
    from toolkit.analysis.utils import get_bins, stack_xarray_dims, concat_stack_xarray_dims
    from toolkit.analysis.signal import gaussian_filter1d_da
    from toolkit.analysis.spikes import smooth_spike_rate
    from toolkit.analysis.statistics import WeightedPCA
    from toolkit.paths.paths import FIGURE_DIR
    from toolkit.plots.format import SAVE_FIGURE, save_figure

    #################### Get session and load data ####################
    session_dir = SessionDirectory(session_id)

    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    session = session_dir.session
    stimulus_presentations = session.stimulus_presentations
    session_type = session.session_type

    stimulus_names = st.STIMULUS_NAMES[session_type]
    drifting_gratings_stimuli = st.STIMULUS_CATEGORIES[session_type]['drifting_gratings'][:1]
    natural_movies_stimuli = st.STIMULUS_CATEGORIES[session_type]['natural_movies']

    layer_of_interest = GLOBAL_SETTINGS['layer_of_interest']
    preferred_orientation = session_dir.load_preferred_orientations().sel(
        layer=[layer_of_interest]).values
    instantaneous_band = GLOBAL_SETTINGS['instantaneous_band']

    #################### Load data and parameters ####################
    # Select combination of stimuli
    match combine_stimulus_name:
        case 'natural_movies':
            stimulus_names = natural_movies_stimuli
        case 'drifting_gratings':
            stimulus_names = drifting_gratings_stimuli
        case 'mixed_stimuli':
            stimulus_names = drifting_gratings_stimuli + natural_movies_stimuli

    stimulus_trials = {}
    for stim in drifting_gratings_stimuli + natural_movies_stimuli:
        stimulus_trials[stim] = st.get_stimulus_trials(stimulus_presentations, stim)

    # Average duration of each frame in natural movies (around 29.97 frames per second)
    bin_width = np.sum([stimulus_trials[stim].duration for stim in natural_movies_stimuli]) \
        / np.sum([stimulus_trials[stim].presentation_increment for stim in natural_movies_stimuli])

    # Select drifting grating conditions
    dg_stim = drifting_gratings_stimuli[0]  # first drifting grating stimulus
    conditions, cond_presentation_id = st.presentation_conditions(stimulus_trials[dg_stim].presentations)
    conditions = ps.filter_conditions(conditions)
    if filter_orientation:
        conditions = conditions.sel(orientation=preferred_orientation)
    cond_presentation_id = {c: cond_presentation_id[c] for c in conditions.values.ravel()}

    # Load LFP power
    lfp_power_dss = session_dir.load_stimulus_lfp_power_combined(instantaneous_band)
    wave_bands = lfp_power_dss[dg_stim].wave_band.values

    # Update valid presentations given available presentations in LFP data (without NaNs)
    valid_presentations = {}
    for stim in drifting_gratings_stimuli + natural_movies_stimuli:
        stimulus_trials[stim] = st.choose_trials(stimulus_trials[stim], lfp_power_dss[stim].presentation_id)
        # Get valid presentations
        valid_presentations[stim] = stimulus_trials[stim].ids  # same as lfp_power_dss[stim].presentation_id

    # Get valid trials with filtered conditions
    for c in list(cond_presentation_id):
        cond_presentation_id[c] = np.intersect1d(cond_presentation_id[c], lfp_power_dss[dg_stim].presentation_id)
    valid_presentations[dg_stim] = np.sort(np.concatenate(list(cond_presentation_id.values())))
    lfp_power_dss[dg_stim] = lfp_power_dss[dg_stim].sel(presentation_id=valid_presentations[dg_stim])

    # Load units info from all sessions
    units_info = FILES.load('all_units_info')
    # Units in current session
    units_info = units_info.loc[units_info['session_id'] == session_id]
    all_units_id = units_info.index.values

    #################### Analyze data ####################
    if sigma == 0:  # If sigma is 0, set to bin width
        sigma = bin_width

    time_dim = 'time_relative_to_stimulus_onset'  # time dimension name for spike counts
    lfp_time_dim = 'time_from_presentation_onset'  # time dimension name for LFP power
    presentation_dim = 'stimulus_presentation_id'  # presentation_id dimension name for spike counts

    # Get spike counts
    if overwrite_spike_counts:
        units_spk_rate = {}
    else:
        units_spk_rate = session_dir.load_units_spike_rate()

    units_frs = {}  # firing rate averaged over trials

    for stim in drifting_gratings_stimuli + natural_movies_stimuli:
        if stim in units_spk_rate:
            spk_counts = units_spk_rate[stim]
        else:
            # Get window from LFP power data and count spikes in bins
            window = lfp_power_dss[stim][lfp_time_dim].values[[0, -1]]
            bin_centers, bin_edges = get_bins(window, bin_width=bin_width, strict_window=True)
            spk_counts = session.presentationwise_spike_counts(
                stimulus_presentation_ids=stimulus_trials[stim].ids, unit_ids=all_units_id, bin_edges=bin_edges)
            spk_counts.coords[time_dim] = bin_centers
            spk_counts = (spk_counts / bin_width).rename('spike_rate')  # convert to firing rate
            spk_counts = spk_counts.assign_attrs(bin_width=bin_width, fs=1 / bin_width, unit='Hz')
            session_dir.save_units_spike_rate({stim: spk_counts})

        # choose valid presentations
        units_spk_rate[stim] = spk_counts.sel(stimulus_presentation_id=valid_presentations[stim])

        # average over trials
        if stim in drifting_gratings_stimuli:
            # select valid presentations
            units_frs[stim] = xr.concat(
                [spk_counts.sel(stimulus_presentation_id=i).mean(dim=presentation_dim) \
                    for i in cond_presentation_id.values()],
                dim=conditions.stack(condition=st.CONDITION_TYPES)
            ).assign_attrs(spk_counts.attrs)
        else:
            units_frs[stim] = spk_counts.mean(dim=presentation_dim).assign_attrs(spk_counts.attrs)

    # Get LFP and power
    # Normalize band power by instantaneous power in layer of interest, downsample to spike rate bin width
    smoothed_band_power = {}
    normalized_band_power = {}
    instantaneous_power = {}
    for stim in stimulus_names:
        lfp_power = lfp_power_dss[stim].sel(layer=layer_of_interest)
        # smooth band power and instantaneous power and normalize
        smoothed_power = gaussian_filter1d_da(lfp_power.band_power, sigma, dim=lfp_time_dim)
        instantaneous = gaussian_filter1d_da(lfp_power.instantaneous_power, sigma, dim=lfp_time_dim)
        # downsample to spike rate bin width
        interp_coord = {lfp_time_dim: units_spk_rate[stim].coords[time_dim]}
        smoothed_power = smoothed_power.interp(interp_coord).assign_attrs(fs=units_spk_rate[stim].attrs['fs'], unit='')
        instantaneous = instantaneous.interp(interp_coord)
        instantaneous_power[stim] = instantaneous = instantaneous.rename(presentation_id=presentation_dim)
        smoothed_band_power[stim] = smoothed_power = smoothed_power.rename(presentation_id=presentation_dim)
        normalized_band_power[stim] = (smoothed_power / instantaneous).assign_attrs(smoothed_power.attrs)

    # Get firing rate statistics
    units_fr = concat_stack_xarray_dims(units_frs.values(), exclude_dims=['unit_id'], reindex=True)
    units_fr = units_fr.to_dataset(name='firing_rate').assign(
        units_fr_mean = units_fr.mean(dim='sample'),
        units_fr_std = units_fr.std(dim='sample'),
        units_fr_min = units_fr.min(dim='sample'),
        units_fr_max = units_fr.max(dim='sample'),
    )

    # Get Soft-normalization parameters (Churchland et al. 2012)
    norm_scale = units_fr['units_fr_' + normalization_scale]
    soft_normalize_cut = np.quantile(norm_scale, quantile)

    # Compute smoothed spike rate of selected units
    # Select only Regular Spiking (RS) units
    unit_idx = np.ones(all_units_id.size, dtype=bool)
    if select_RS:
        unit_idx = units_info['unit_type'] == 'RS'
    if select_layer:
        if select_layer is True:
            select_layer = [layer_of_interest]
        unit_idx = unit_idx & units_info['layer_acronym'].isin(select_layer)

    units_id = all_units_id[unit_idx]
    n_units = units_id.size
    print(f"Number of RS units: {n_units}/{all_units_id.size}")

    # Determine config name
    config_name = combine_stimulus_name
    config_name += '_orient' if filter_orientation else ''
    config_name += '_RS_units' if select_RS else '_all_units'
    config_name += '_layer_' + '_'.join(select_layer) if select_layer else '_all_layers'
    config_name += '_' + config_name_suffix if config_name_suffix else ''

    # Check if selected enough units
    if n_units <= n_pc_range[0] + 1:
        session_parameters = dict(
            soft_normalize_cut=soft_normalize_cut,
            n_units=n_units
        )
        safe_mkdir(FILES.population_vector_dir(config_name))
        session_dir.save_population_vector_parameters(config_name, session_parameters)
        raise ValueError(f"Insufficient number of units ({n_units:d}) for PCA analysis.")

    # Compute smoothed and trial-averaged spike rates
    units_smoothed_rate = {}
    units_averaged_rate = {}
    for stim in stimulus_names:
        units_smoothed_rate[stim] = smooth_spike_rate(
            units_spk_rate[stim].sel(unit_id=units_id), sigma,
            normalization_scale=norm_scale.sel(unit_id=units_id) if normalize_unit_fr else None,
            soft_normalize_cut=soft_normalize_cut if soft_normalize else 0,
            time_dim=time_dim
        )
        units_averaged_rate[stim] = units_frs[stim].sel(unit_id=units_id)

    # PCA analysis
    n_samples_stim = {}  # total number of trial-averaged time samples per stimulus
    n_trials_stim = {}  # average number of trials per stimulus condition
    for stim in stimulus_names:
        n_samples_stim[stim] = units_averaged_rate[stim].size // n_units
        n_trials_stim[stim] = units_smoothed_rate[stim].size / units_averaged_rate[stim].size

    # Concatenate trial-averaged firing rate samples
    units_vec = concat_stack_xarray_dims(units_averaged_rate.values(), exclude_dims=['unit_id']).transpose('sample', 'unit_id')
    # Get weights by the number of trials
    trial_weights = np.concatenate([np.full(n, n_trials_stim[stim]) for stim, n in n_samples_stim.items()])

    pca = WeightedPCA(n_components=n_units)
    pca.fit(units_vec, weights=trial_weights)

    # Determine number of main principal components as maximum explained variance drop
    ev = pca.explained_variance_ratio_
    ev_drop = ev[1:] / ev[:-1]
    n_main_pc = np.argmin(ev_drop[n_pc_range[0] - 1:n_pc_range[1]]).item() + n_pc_range[0]

    print(f"Explained variance % of top {n_pc_range[1] + 1} components: \n" + 
        ', '.join('%.2f' % (100 * x) for x in ev[:n_pc_range[1] + 1]))
    print(f"Explained variance % drop of top {n_pc_range[1] + 1} components: \n" + 
        ', '.join('%.1f' % (100 * (1 - x)) for x in ev_drop[:n_pc_range[1]]))

    # Get population vectors from PCA
    sample_dims = (presentation_dim, time_dim)

    if absolute_origin:
        pop_vec_org = pca.transform(np.zeros([1, n_units]))[0]
    else:
        pop_vec_org = np.zeros(n_units)
    pop_vec_org = xr.DataArray(pop_vec_org, coords=dict(PC=range(n_units)))

    pop_vecs = {}
    for stim, da in units_smoothed_rate.items():
        # concatenate time samples
        pop_vec = stack_xarray_dims(da, dims=sample_dims, create_index=True).transpose('sample', 'unit_id')
        # transform to principal components and recover time dimensions for per trial calculation
        pop_vec = pop_vec.copy(data=pca.transform(pop_vec)).rename(unit_id='PC').unstack('sample')
        # set absolute origin
        pop_vecs[stim] = pop_vec.assign_coords(PC=range(n_units)) - pop_vec_org

    # Calculate variables of population activities
    var_df = pd.DataFrame.from_dict(dict(
        mean_fr=dict(data={}, label='Mean firing rate', fmt='{:.2f}'),
        rms_fr=dict(data={}, label=f'RMS firing rate', fmt='{:.2f}'),
        length=dict(data={}, label='Magnitude of population vector', fmt='{:.2f}'),
        speed=dict(data={}, label='Magnitude of population change', fmt='{:.2f}'),
        angle=dict(data={}, label='Angle of population change (degree)', fmt='{:.0f}'),
        radial=dict(data={}, label='Population radial change', fmt='{:.2f}'),
        tangent=dict(data={}, label='Magnitude of population tangent change', fmt='{:.2f}')
    ), orient='index')
    var_df.index.name = 'variable_name'
    var_col = var_df['data']

    # time indices per trial
    t0_idx = {time_dim: slice(1, None)}  # current time indices
    t1_idx = {time_dim: slice(None, -1)}  # previous time indices

    # average firing rate across units
    for stim, da in units_smoothed_rate.items():
        frs = da.isel(t0_idx)
        var_col['mean_fr'][stim] = frs.mean(dim='unit_id')  # mean firing rate
        var_col['rms_fr'][stim] = np.sqrt((frs ** 2).mean(dim='unit_id'))  # root mean square firing rate

    for stim, pop_vec in pop_vecs.items():
        pop_vec = pop_vec.sel(PC=range(n_main_pc))  # consider only main PCs
        pop_vec0 = pop_vec.isel(t0_idx)  # current vectors
        pop_vec1 = pop_vec.isel(t1_idx)  # previous vectors
        t = pop_vec0.coords[time_dim]  # current time points
        pop_vec1 = pop_vec1.assign_coords({time_dim: t})  # align time coordinates
        pop_vec_len = vec_len(pop_vec)  # vector length
        pop_vec0_len = pop_vec_len.isel(t0_idx)
        pop_vec1_len = pop_vec_len.isel(t1_idx).assign_coords({time_dim: t})
        pop_vel = pop_vec0 - pop_vec1  # velocity
        unit_vec0, unit_vec1 = pop_vec0 / pop_vec0_len, pop_vec1 / pop_vec1_len  # unit vectors

        var_col['length'][stim] = pop_vec0_len
        var_col['speed'][stim] = vec_len(pop_vel)
        var_col['angle'][stim] = np.degrees(np.arccos(np.clip(vec_dot(unit_vec0, unit_vec1), -1, 1)))
        var_col['radial'][stim] = radial_vel = vec_dot(pop_vel, unit_vec0)
        var_col['tangent'][stim] = vec_len(pop_vel - radial_vel * unit_vec0)

    # concatenate time samples for each variable
    for d in var_col.index:
        var_col[d] = concat_stack_xarray_dims(var_col[d].values(), dims=sample_dims)
    # concatenate time samples for lfp band power at aligned time points
    power_ds = concat_stack_xarray_dims([da.isel(t0_idx) for da in normalized_band_power.values()], dims=sample_dims)
    # Construct dataset with lfp band power and population variables
    power_ds = xr.Dataset(
        data_vars=dict(power=power_ds, **var_col, **var_df[['label', 'fmt']].to_xarray()),
        attrs=dict(stimulus=combine_stimulus_name)
    )

    #################### Save results ####################
    global_parameters = dict(
        units_firing_rate = dict(
            sigma=sigma,
            normalize_unit_fr=normalize_unit_fr,
            soft_normalize=soft_normalize,
            normalization_scale=normalization_scale,
            quantile=quantile
        ),
        pca=dict(
            n_pc_range=n_pc_range,
            absolute_origin=absolute_origin
        ),
        stimuli_selection=dict(
            combine_stimulus_name=combine_stimulus_name,
            filter_orientation=filter_orientation
        ),
        units_selection=dict(
            select_RS=select_RS,
            select_layer=select_layer
        )
    )

    session_parameters = dict(
        soft_normalize_cut=soft_normalize_cut,
        n_units=n_units,
        n_main_pc=n_main_pc,
        n_samples_stim=n_samples_stim,
        n_trials_stim=n_trials_stim
    )

    # Save global parameters (only by main job to avoid race condition)
    safe_mkdir(FILES.population_vector_dir(config_name))
    if is_main_job:
        with open(FILES.population_vector_parameters(config_name), 'w') as f:
            json.dump(global_parameters, f, indent=4)

    # Save session parameters
    session_dir.save_population_vector_parameters(config_name, session_parameters)

    # Save data
    session_dir.save_population_vector_data(config_name, power_ds)


    #################### Save figures ####################
    if not SAVE_FIGURE:
        return

    import matplotlib.pyplot as plt
    from toolkit.plots.plots import plot_multicolor_line
    from toolkit.plots.colors import VISP_LAYER_COLORS_DICT
    from collections import defaultdict

    fig_dir = FIGURE_DIR / "population_vector"
    config_dir = fig_dir / config_name
    firing_stats_dir = fig_dir / "units_firing_stats"
    pca_dir = config_dir / "pca_explained_variance"
    trajectory_dir = config_dir / "average_trajectory"
    scatter_dir = config_dir / "all_trials_scatter"
    safe_mkdir(firing_stats_dir)
    safe_mkdir(pca_dir)
    safe_mkdir(scatter_dir)

    session_str = f"session_{session_id}"

    PC_disp = [0, 1]
    figsize = (12, 5)
    cmap = 'jet'

    # Plot units firing statistics
    n_all_units = len(units_info)
    unit_layer = units_info['layer_acronym'].values
    layer_boundary_idx = np.nonzero(unit_layer[:-1] != unit_layer[1:])[0]
    layer_boundary_idx = np.hstack([-1, layer_boundary_idx, unit_layer.size - 1])
    layer_colors = defaultdict(lambda: 'gray', VISP_LAYER_COLORS_DICT)  # if outside VISP, use gray
    is_RS = units_info['unit_type'].values == 'RS'
    positions = np.arange(n_all_units)
    boundary_gap = 0.1
    firing_rate = units_fr.firing_rate.values.T  # (samples, units)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.violinplot(firing_rate, positions=positions, widths=.9)
    ax.boxplot(firing_rate, positions=positions, widths=0.4, whis=(0, 100),
        showmeans=True, meanline=True, showfliers=False, showcaps=False,
        meanprops={'linestyle': '-'}, whiskerprops={'color':'none'}, boxprops={'color': 'blue'})
    for i in layer_boundary_idx[:-1]:
        clr = layer_colors[unit_layer[i + 1]]
        ax.axvline(i + 0.5 + boundary_gap, linestyle='--', color=clr)
        ax.annotate(unit_layer[i + 1], xy=(i + 0.5 + boundary_gap, ax.get_ylim()[1]),
            xytext=(5, -5), textcoords='offset points', ha='left', va='top', color=clr)
    for i in layer_boundary_idx[1:]:
        ax.axvline(i + 0.5 - boundary_gap, linestyle='--', color=layer_colors[unit_layer[i]])
    ax.set_xticks(positions, labels=units_info['unit_type'], rotation=75)
    for lb in ax.get_xticklabels():
        lb.set_color('tab:blue' if lb.get_text() == 'RS' else 'tab:red')
    ax.set_xlabel('units')
    ax.set_ylabel('firing rate (Hz)')

    ax = fig.add_subplot(2, 2, 3)
    ax.axhline(soft_normalize_cut, linestyle='--', color='orange')
    ax.plot(units_fr.units_fr_max, units_fr.units_fr_min, 'm.', marker='_', markersize=8, label='min')
    ax.plot(units_fr.units_fr_max[is_RS], units_fr.units_fr_std[is_RS], 'b.', label='std')
    ax.plot(units_fr.units_fr_max[~is_RS], units_fr.units_fr_std[~is_RS], 'bo', markerfacecolor='none')
    ax.plot(units_fr.units_fr_max[is_RS], units_fr.units_fr_mean[is_RS], 'r.', label='mean')
    ax.plot(units_fr.units_fr_max[~is_RS], units_fr.units_fr_mean[~is_RS], 'ro', markerfacecolor='none')
    ax.plot([], [], 'k.', label='RS units')
    ax.plot([], [], 'ko', markerfacecolor='none', label='non-RS units')
    ax.set_xlabel('unit max firing rate (Hz)')
    ax.set_ylabel('Hz')
    ax.legend(loc='upper left', framealpha=0.2, fontsize='small')

    ax = fig.add_subplot(2, 2, 4)
    ax.hist([norm_scale[is_RS], norm_scale[~is_RS]], bins=30, stacked=True,
        color=['tab:blue', 'tab:red'], label=['RS units', 'non-RS units'])
    ax.axvline(soft_normalize_cut, linestyle='--', color='orange', label='soft_normalize_cut')
    ax.set_xlabel(f'Unit firing rate {normalization_scale:s} (Hz)')
    ax.set_ylabel('Count')
    ax.legend(loc='upper right', framealpha=0.2, fontsize='small')

    fig.suptitle('Units trial-averaged firing rate across conditions')
    fig.tight_layout()

    save_figure(firing_stats_dir, fig, name=session_str)

    # Plot PCA explained variance
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.6))
    ax.bar(np.arange(n_units) + 1, 100 * ev)
    yl = ax.get_ylim()[1]
    ax.plot(np.arange(n_units - 1) + 1.3, 100 * (1 - ev_drop), 'gray', label='Percent drop')
    ax.axvline(n_pc_range[0] - 0.5, linestyle='--', color='gray', label='Number of PCs range')
    ax.axvline(n_pc_range[1] + 0.5, linestyle='--', color='gray')
    ax.axvline(n_main_pc + 0.5, color='r', label=f'{n_main_pc} main PCs (maximum drop)')
    ax.set_xlim(0, min(n_units, 50) + 1)
    ax.set_ylim(0, yl)
    ax.set_xlabel('Principal components')
    ax.set_ylabel('Explained variance (%)')
    ax.legend(loc='upper right', framealpha=0.2, fontsize='small')
    fig.tight_layout()
    save_figure(pca_dir, fig, name=session_str)

    # Plot average trajectory
    for stim, pop_vec in pop_vecs.items():
        stim_dir = trajectory_dir / stim
        safe_mkdir(stim_dir)
        pop_vec_avg = pop_vec.sel(PC=PC_disp).mean(dim=presentation_dim).transpose('PC', time_dim)
        power = normalized_band_power[stim].isel(t0_idx).mean(dim=presentation_dim)
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        for ax, w in zip(axs, wave_bands):
            line, _ = plot_multicolor_line(*pop_vec_avg, c=power.sel(wave_band=w), ax=ax, cmap=cmap)
            ax.plot(*pop_vec_avg.sel({time_dim: 0.}, method='nearest'),
                marker='o', markeredgecolor ='g', markersize=12, markeredgewidth=3, markerfacecolor='none')
            ax.plot(*pop_vec_avg.sel({time_dim: stimulus_trials[stim].duration}, method='nearest'),
                marker='x', markeredgecolor ='darkred', markersize=12, markeredgewidth=3, markerfacecolor='none')
            plt.colorbar(mappable=line, ax=ax, label='Normalized power', shrink=0.6)
            ax.autoscale_view()
            ax.set_xlabel(f'PC {PC_disp[0]:d}')
            ax.set_ylabel(f'PC {PC_disp[1]:d}')
            ax.set_title(w)
        fig.suptitle(stim + ' average trajectory')
        fig.tight_layout()
        save_figure(stim_dir, fig, name=session_str)

    # Plot population vector all trials scatter
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    pop_vec_da = concat_stack_xarray_dims([da.sel(PC=PC_disp).isel(t0_idx) for da in pop_vecs.values()],
        dims=sample_dims).transpose('PC', 'sample')
    for ax, w in zip(axs, wave_bands):
        sc = ax.scatter(*pop_vec_da, c=power_ds.power.sel(wave_band=w), s=1, marker='.', cmap=cmap)
        plt.colorbar(mappable=sc, ax=ax, label='Normalized power', shrink=0.6)
        ax.autoscale_view()
        ax.set_xlabel(f'PC {PC_disp[0]:d}')
        ax.set_ylabel(f'PC {PC_disp[1]:d}')
        ax.set_title(w)
    fig.suptitle(f"All trials in {power_ds.attrs['stimulus']:s}")
    fig.tight_layout()
    save_figure(scatter_dir, fig, name=session_str)

    plt.close('all')


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    # Determine if running as main job
    array_index = args['array_index']
    args['parameters']['is_main_job'] = array_index is None or array_index == 0

    process_sessions(population_vector_during_stimuli, **args)
