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
):
    import numpy as np
    import pandas as pd
    import xarray as xr
    from matplotlib import pyplot as plt

    import add_path
    import toolkit.allen_helpers.stimuli as st
    import toolkit.pipeline.signal as ps
    from toolkit.pipeline.data_io import SessionDirectory
    from toolkit.analysis.signal import bandpass_power
    from toolkit.plots.plots import plot_channel_signal_array
    from toolkit.paths import RESULTS_DIR, FIGURE_DIR
    from toolkit.plots.format import SAVE_FIGURE, save_figure
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS

    #################### Get session and probe ####################
    session_dir = SessionDirectory(session_id, cache_lfp=True)

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
        layer=GLOBAL_SETTINGS['layer_of_interest']).item()

    # load bands of interest
    bands_of_interest = xr.load_dataarray(RESULTS_DIR / f'bands_of_interest.nc')

    # parameters
    wave_bands = bands_ds.wave_band.values

    csd_plot_kwargs = dict(
        central_channels=central_channels,
        ccf_coordinates=lfp_channels['dorsal_ventral_ccf_coordinate']
    )


    #################### Analyze data and save figures ####################
    # Figures directory
    session_str = f"session_{session_id}"

    fig_dir = FIGURE_DIR / "CSD"
    flashes_dir = fig_dir / "flashes"
    drifting_gratings_dir = fig_dir / "drifting_gratings"
    flashes_dir.mkdir(parents=True, exist_ok=True)
    drifting_gratings_dir.mkdir(parents=True, exist_ok=True)


    # Flashes CSD
    stim = 'flashes'
    flashes_trials = st.get_stimulus_trials(stimulus_presentations, stim)
    flashes_blocks = st.get_stimulus_blocks(flashes_trials)

    window = (flashes_window[0], flashes_trials.duration + flashes_window[1])
    aligned_csd = st.align_trials_from_blocks(csd_array, flashes_blocks, window=window)[0]
    average_csd = aligned_csd.mean(dim='presentation_id', keep_attrs=True)
    total_power = (aligned_csd ** 2).mean(dim=('presentation_id', 'time_from_presentation_onset'))

    freq_bands = {}
    for wave_band in wave_bands:
        freq_band = ps.get_band_with_highest_peak(bands_ds.sel(stimulus=stim, wave_band=wave_band))
        if freq_band is not None:
            freq_bands[wave_band] = freq_band.values

    aligned_csd_power = {}
    for wave_band, freq_band in freq_bands.items():
        csd_power = bandpass_power(ps.bandpass_filter_blocks(
            csd_array, flashes_blocks, freq_band,
            extend_time=extend_time,
            include_filtered=False,
            include_amplitude=True
        ))

        aligned_csd_power[wave_band] = st.align_trials_from_blocks(
            csd_power, flashes_blocks, window=window
        )[0].mean(dim='presentation_id', keep_attrs=True)

    # Average CSD
    ax = plot_channel_signal_array(
        average_csd.time_from_presentation_onset, channel_positions,
        average_csd, clabel='CSD', **csd_plot_kwargs
    )
    ax.axvline(0, color='w')
    ax.axvline(flashes_trials.duration, color='w')
    ax.set_title(f'{stim} average CSD')

    save_figure(flashes_dir, ax.get_figure(), name=f'{session_str}_average')

    # CSD power
    fig, axs = plt.subplots(len(freq_bands), 1, figsize=(6.4, 4.8 * len(freq_bands)),
        sharex=True, squeeze=False)
    for ax, (wave_band, csd_power) in zip(axs.ravel(), aligned_csd_power.items()):
        plot_channel_signal_array(
            csd_power.time_from_presentation_onset, channel_positions,
            csd_power / total_power, clabel='Normalized CSD power', **csd_plot_kwargs, ax=ax
        )
        ax.axvline(0, color='w')
        ax.axvline(flashes_trials.duration, color='w')
        band = freq_bands[wave_band]
        ax.set_title(f'{wave_band} ({band[0]:.1f} - {band[1]:.1f} Hz)')
    fig.suptitle(f'{stim} CSD power')
    fig.tight_layout()

    save_figure(flashes_dir, fig, name=f'{session_str}_power')


    # Drifting gratings CSD
    stim = drifting_gratings_stimuli[0]  # first drifting grating stimulus
    drifting_gratings_trials = st.get_stimulus_trials(stimulus_presentations, stim)
    drifting_gratings_blocks = st.get_stimulus_blocks(drifting_gratings_trials)

    window = (drifting_gratings_window[0], drifting_gratings_trials.duration + drifting_gratings_window[1])
    aligned_csd = st.align_trials_from_blocks(csd_array, drifting_gratings_blocks, window=window)[0]
    average_csd = aligned_csd.mean(dim='presentation_id', keep_attrs=True)
    total_power = (aligned_csd ** 2).mean(dim=('presentation_id', 'time_from_presentation_onset'))


