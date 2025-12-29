"""
Analyze CSD.

- Get CSD of stimuli flashes and drifting gratings.
- Get trial averaged CSD and CSD power in wave bands from FOOOF results.
- Save figures.
"""

import add_path

PARAMETERS = dict(
    extend_time = dict(
        default = 1.0,
        type = float,
        help = "Extend time at the start and end of each block to avoid boundary effect for filtering."
    ),
    flashes_window = dict(
        default = [-0.2, 0.75],
        type = list,
        help = "Extend window for flashes CSD from the start and end of each trial."
    ),
    drifting_gratings_window = dict(
        default = [-0.5, 0.5],
        type = list,
        help = "Extend window for drifting gratings CSD from the start and end of each trial."
    )
)


def analyze_csd(
    session_id: int,
    extend_time: float,
    flashes_window: list[float, float],
    drifting_gratings_window: list[float, float],
) -> None:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from matplotlib import pyplot as plt

    import toolkit.allen_helpers.stimuli as st
    import toolkit.pipeline.signal as ps
    from toolkit.pipeline.data_io import SessionDirectory
    from toolkit.analysis.signal import bandpass_power
    from toolkit.plots.plots import plot_channel_signal_array
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS
    from toolkit.paths import RESULTS_DIR, FIGURE_DIR
    from toolkit.plots.format import SAVE_FIGURE, save_figure

    #################### Get session and probe ####################
    session_dir = SessionDirectory(session_id)

    probe_info = session_dir.load_probe_info()
    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    session = session_dir.session
    stimulus_presentations = session.stimulus_presentations
    session_type = session.session_type
    drifting_gratings_stimuli = st.STIMULUS_CATEGORIES[session_type]['drifting_gratings']

    #################### Load data and parameters ####################
    central_channels = probe_info['central_channels']

    lfp_channels = session_dir.load_lfp_channels()
    channel_positions = lfp_channels['probe_vertical_position']

    csd_array = session_dir.load_csd()

    bands_ds = session_dir.load_wave_bands()
    preferred_orientation = session_dir.load_preferred_orientations().sel(
        layer=[GLOBAL_SETTINGS['layer_of_interest']]).values

    # load bands of interest
    bands_of_interest = xr.load_dataarray(RESULTS_DIR / f'bands_of_interest.nc')

    # parameters
    wave_bands = bands_ds.wave_band.values

    coordinates = lfp_channels['dorsal_ventral_ccf_coordinate']
    if np.isnan(coordinates).any():  # if ccf coordinates are missing, use probe vertical position
        coordinates = lfp_channels['probe_vertical_position']
        coordinates_label = 'Vertical Position'
    else:
        coordinates_label = 'Dorsal-Ventral CCF'

    csd_plot_kwargs = dict(
        channel_positions=channel_positions,
        central_channels=central_channels,
        coordinates=coordinates,
        coordinates_label=coordinates_label
    )


    #################### Analyze data and save figures ####################
    csd_dss = {}

    # Flashes
    stim = 'flashes'
    stimulus_trials = st.get_stimulus_trials(stimulus_presentations, stim)
    stimulus_blocks = st.get_stimulus_blocks(stimulus_trials)

    window = (flashes_window[0], stimulus_trials.duration + flashes_window[1])
    aligned_csd = st.align_trials_from_blocks(csd_array, stimulus_blocks, window=window, ignore_nan_trials='any')[0]
    average_csd = aligned_csd.mean(dim='presentation_id', keep_attrs=True)
    total_power = (aligned_csd ** 2).mean(dim=('presentation_id', 'time_from_presentation_onset'))

    freq_bands = []
    for wave_band in wave_bands:
        freq_band = ps.get_band_with_highest_peak(bands_ds.sel(stimulus=stim, wave_band=wave_band))
        if freq_band is not None:
            freq_bands.append(freq_band)
    freq_bands = xr.concat(freq_bands, dim='wave_band', coords=['layer'])

    csd_power = {}
    for wave_band in freq_bands.wave_band.values:
        block_power = bandpass_power(ps.bandpass_filter_blocks(
            csd_array, stimulus_blocks,
            freq_bands.sel(wave_band=wave_band).values,
            extend_time=extend_time,
            include_filtered=False,
            include_amplitude=True
        ))

        csd_power[wave_band] = st.align_trials_from_blocks(
            block_power, stimulus_blocks, window=window
        )[0].mean(dim='presentation_id', keep_attrs=True)
    csd_power = xr.concat(csd_power.values(), dim=pd.Index(csd_power, name='wave_band'))

    csd_dss['flashes'] = xr.Dataset(dict(
            average=average_csd,
            power=csd_power,
            total_power=total_power,
            bands=freq_bands,
            is_band_of_interest=('wave_band', np.full(freq_bands.wave_band.size, False))
        ),
        attrs=average_csd.attrs | dict(
            duration=stimulus_trials.duration,
            gap_duration=stimulus_trials.gap_duration
        )
    )

    # Drifting gratings
    stim = drifting_gratings_stimuli[0]  # first drifting grating stimulus
    stimulus_trials = st.get_stimulus_trials(stimulus_presentations, stim)
    stimulus_blocks = st.get_stimulus_blocks(stimulus_trials)
    conditions = st.presentation_conditions(stimulus_trials.presentations)

    window = (drifting_gratings_window[0], stimulus_trials.duration + drifting_gratings_window[1])
    aligned_csd, valid_trials = st.align_trials(
        csd_array, stimulus_trials, window=window, ignore_nan_trials='any')

    if valid_trials is not None:  # if any trial is dropped by NaN values
        cond_presentation_id = st.presentation_conditions(valid_trials.presentations)[1]
        if len(conditions[1]) != len(cond_presentation_id):
            diff = set(conditions[1].keys()) - set(cond_presentation_id.keys())
            raise ValueError(f"All trials are dropped by NaN values in {stim} for conditions: {diff}")
        conditions = (conditions[0], cond_presentation_id)

    # average across presentations of same condition, select conditions, average across conditions (and time)
    average_csd = ps.filter_conditions(st.average_trials_with_conditions(aligned_csd, *conditions)) \
        .sel(orientation=preferred_orientation).mean(dim=st.CONDITION_TYPES, keep_attrs=True)

    total_power = ps.filter_conditions(st.average_trials_with_conditions(aligned_csd ** 2, *conditions)) \
        .sel(orientation=preferred_orientation).mean(dim=(*st.CONDITION_TYPES, 'time_from_presentation_onset'))

    freq_bands = []
    is_band_of_interest = []
    for wave_band in wave_bands:
        # check if wave band is already extracted from layer of interest
        is_band_of_interest_ = wave_band in bands_of_interest.wave_band and session_id in bands_of_interest.session_id
        if is_band_of_interest_:
            freq_band = bands_of_interest.sel(wave_band=wave_band, session_id=session_id)
        else:  # fallback to extract from all layers
            freq_band = ps.get_band_with_highest_peak(bands_ds.sel(stimulus=stim, wave_band=wave_band))
        if freq_band is not None:
            freq_bands.append(freq_band)
            is_band_of_interest.append(is_band_of_interest_)
    freq_bands = xr.concat(freq_bands, dim='wave_band', coords=['layer'])  # also concatenate coordinate 'layer'

    csd_power = {}
    for wave_band in freq_bands.wave_band.values:
        block_power = bandpass_power(ps.bandpass_filter_blocks(
            csd_array, stimulus_blocks,
            freq_bands.sel(wave_band=wave_band).values,
            extend_time=extend_time,
            include_filtered=False,
            include_amplitude=True
        ))
        block_power, _ = st.align_trials(
            block_power, stimulus_trials if valid_trials is None else valid_trials,
            window=window, ignore_nan_trials=''
        )

        csd_power[wave_band] = ps.filter_conditions(st.average_trials_with_conditions(block_power, *conditions)) \
            .sel(orientation=preferred_orientation).mean(dim=st.CONDITION_TYPES, keep_attrs=True)
    csd_power = xr.concat(csd_power.values(), dim=pd.Index(csd_power, name='wave_band'))

    csd_dss['drifting_gratings'] = xr.Dataset(dict(
            average=average_csd,
            power=csd_power,
            total_power=total_power,
            bands=freq_bands,
            is_band_of_interest=('wave_band', is_band_of_interest)
        ),
        attrs=average_csd.attrs | dict(
            duration=stimulus_trials.duration,
            gap_duration=stimulus_trials.gap_duration
        )
    )


    #################### Save data ####################
    session_dir.save_stimulus_csd(csd_dss)
    

    #################### Save figures ####################
    if not SAVE_FIGURE:
        return

    fig_dir = FIGURE_DIR / "CSD"

    session_str = f"session_{session_id}"

    # Plot CSD
    for stim, csd_ds in csd_dss.items():
        stim_dir = fig_dir / stim
        stim_dir.mkdir(parents=True, exist_ok=True)

        # average CSD
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        plot_channel_signal_array(time=csd_ds.time_from_presentation_onset,
            signal=csd_ds.average, clabel='CSD', **csd_plot_kwargs, ax=ax)
        ax.axvline(0, color='w')
        ax.axvline(csd_ds.attrs['duration'], color='w')
        ax.set_title(f'{stim} average CSD')
        fig.tight_layout()

        save_figure(stim_dir, fig, name=f'{session_str}_average')

        # CSD power
        n_bands = csd_ds.wave_band.size
        freq_bands = csd_ds.bands

        fig, axs = plt.subplots(n_bands, 1, figsize=(6.4, 4.0 * n_bands), sharex=True, squeeze=False)
        for ax, wave_band in zip(axs.ravel(), csd_ds.wave_band.values):
            plot_channel_signal_array(
                time=csd_ds.time_from_presentation_onset,
                signal=csd_ds.power.sel(wave_band=wave_band) / csd_ds.total_power,
                clabel='Normalized CSD power', **csd_plot_kwargs, ax=ax
            )
            ax.axvline(0, color='w')
            ax.axvline(csd_ds.attrs['duration'], color='w')

            # title info
            band = csd_ds.bands.sel(wave_band=wave_band).values
            layer = csd_ds.layer.sel(wave_band=wave_band).item()
            of_interest = ' of interest' if csd_ds.is_band_of_interest.sel(wave_band=wave_band).item() else ''
            ax.set_title(f'{wave_band} ({band[0]:.1f} - {band[1]:.1f} Hz in layer {layer}{of_interest})')
        fig.suptitle(f'{stim} CSD power')
        fig.tight_layout()

        save_figure(stim_dir, fig, name=f'{session_str}_power')

    plt.close('all')  # free memory


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(analyze_csd, **args)
