"""
LFP power during stimuli.

- Get beta/gamma frequency bands from drifting gratings and natural movies.
- Compute LFP power (instantaneous power and band power) during drifting gratings and natural movies.
- Save data.
"""

import add_path

import numpy as np
import toolkit.pipeline.signal as ps


PARAMETERS = dict(
    group_width = dict(
        default = 1,
        type = int,
        help = "Number of channels to the left and right of the central channel to average LFP over."
    ),
    extend_time = dict(
        default = 1.0,
        type = float,
        help = "Extend time at the start and end of each block to avoid boundary effect for filtering."
    ),
    filter_instantaneous_power = dict(
        default = True,
        type = bool,
        help = "Whether to filter before calculating instantaneous power."
    ),
    drifting_gratings_window = dict(
        default = [-0.5, 0.5],
        type = list,
        help = "Extend window for drifting gratings LFP from the start and end of each trial."
    ),
    natural_movies_window = dict(
        default = [0., 0.],
        type = list,
        help = "Extend window for natural movies LFP from the start and end of each trial."
    )
)


def find_band_in_layers(bands_ds, layer_of_interest):
    band = bands_ds.bands.sel(layer=layer_of_interest)
    if np.isnan(band).all():
        wave_band = bands_ds.wave_band.item().title()
        print(f"{wave_band} band not found in the layer of interest '{layer_of_interest}'. "
            "Trying to find them in other layers.")
        band = ps.get_band_with_highest_peak(bands_ds)
        if band is None:
            print(f"{wave_band} band not found in any layer.")
        else:
            print(f"{wave_band} band found in layer '{band.layer.item()}'.")
    return band


def lfp_power_during_stimuli(
    session_id: int,
    group_width: int = 1,
    extend_time: float = 1.0,
    filter_instantaneous_power: bool = True,
    drifting_gratings_window: list[float, float] = [-0.5, 0.5],
    natural_movies_window: list[float, float] = [0., 0.],
) -> None:
    import pandas as pd
    import xarray as xr

    import toolkit.allen_helpers.stimuli as st
    from toolkit.analysis.signal import bandpass_power, instantaneous_power
    from toolkit.pipeline.data_io import SessionDirectory, FILES
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
    natural_movies_stimuli = st.STIMULUS_CATEGORIES[session_type]['natural_movies']

    #################### Load data and parameters ####################
    wave_bands = ['beta', 'gamma']
    bands_of_interest = FILES.load('bands_of_interest')
    layer_of_interest = GLOBAL_SETTINGS['layer_of_interest']
    instantaneous_band = GLOBAL_SETTINGS['instantaneous_band']

    # Get frequency bands of interest
    if session_id in bands_of_interest.session_id:
        freq_bands = bands_of_interest.sel(wave_band=wave_bands, session_id=session_id)
    else:
        print("Warning: Bands of interest not found in the PSD of this session. "
            f"Trying to find them in layers other than the layer of interest '{layer_of_interest}'.")
        bands_ds = session_dir.load_wave_bands()
        beta_stim = drifting_gratings_stimuli[0] + '_filtered'
        gamma_stim = natural_movies_stimuli[0]
        beta_band = find_band_in_layers(bands_ds.sel(stimulus=beta_stim, wave_band='beta'), layer_of_interest)
        gamma_band = find_band_in_layers(bands_ds.sel(stimulus=gamma_stim, wave_band='gamma'), layer_of_interest)
        if beta_band is None or gamma_band is None:
            average_wave_bands = FILES.load('average_wave_bands', session_type=session_type, session_set='selected_sessions')
            if beta_band is None:
                print("Beta band not found in the session. Trying to find the average bands from selected sessions.")
                beta_band = find_band_in_layers(average_wave_bands.sel(stimulus=beta_stim, wave_band='beta'), layer_of_interest)
                if beta_band is None:
                    raise ValueError("No beta band found.")
            if gamma_band is None:
                print("Gamma band not found in the session. Trying to find the average bands from selected sessions.")
                gamma_band = find_band_in_layers(average_wave_bands.sel(stimulus=gamma_stim, wave_band='gamma'), layer_of_interest)
                if gamma_band is None:
                    raise ValueError("No gamma band found.")
        freq_bands = xr.concat([beta_band, gamma_band], dim='wave_band')

    # Get LFP groups
    lfp_groups, _ = ps.get_lfp_channel_groups(session_dir,
        probe_info['central_channels'], probe_id=probe_info['probe_id'], width=group_width)
    session_dir.clear_lfp_cache()  # clear to avoid memory leak

    #################### Analyze data ####################
    lfp_power_dss = {}

    # Drifting gratings
    stim = drifting_gratings_stimuli[0]  # first drifting grating stimulus
    stimulus_trials = st.get_stimulus_trials(stimulus_presentations, stim)
    conditions = st.presentation_conditions(stimulus_trials.presentations)
    window = (drifting_gratings_window[0], stimulus_trials.duration + drifting_gratings_window[1])
    aligned_lfp, valid_trials = st.align_trials(
        lfp_groups, stimulus_trials, window=window, ignore_nan_trials='any')

    if valid_trials is None:
        valid_trials = stimulus_trials
    else:  # if any trial is dropped by NaN values
        cond_presentation_id = st.presentation_conditions(valid_trials.presentations)[1]
        if len(conditions[1]) != len(cond_presentation_id):
            diff = set(conditions[1].keys()) - set(cond_presentation_id.keys())
            raise ValueError(f"All trials are dropped by NaN values in {stim} for conditions: {diff}")
        conditions = (conditions[0], cond_presentation_id)
    valid_blocks = st.get_stimulus_blocks(valid_trials)

    lfp_bands_power = []
    for wave_band in wave_bands:
        block_power = [bandpass_power(x) for x in ps.bandpass_filter_blocks(
            lfp_groups, valid_blocks,
            freq_bands.sel(wave_band=wave_band).values,
            extend_time=extend_time,
            include_filtered=False,
            include_amplitude=True,
            concat=False
        )]
        lfp_bands_power.append(st.align_trials_from_blocks(block_power, valid_blocks, window=window, ignore_nan_trials='')[0])
    lfp_bands_power = xr.concat(lfp_bands_power, dim=pd.Index(wave_bands, name='wave_band'), combine_attrs='drop_conflicts')

    if filter_instantaneous_power:
        block_filt = [x.filtered.assign_attrs(x.attrs) for x in ps.bandpass_filter_blocks(
            lfp_groups, valid_blocks, instantaneous_band, extend_time=extend_time, concat=False)]
        aligned_lfp = st.align_trials_from_blocks(block_filt, valid_blocks, window=window, ignore_nan_trials='')[0]
    lfp_power_dss[stim] = xr.Dataset(
        data_vars = dict(
            instantaneous_power = instantaneous_power(aligned_lfp),
            band_power = lfp_bands_power,
            freq_bands = freq_bands,
        )
    )

    # Natural movies
    for stim in natural_movies_stimuli:
        stimulus_trials = st.get_stimulus_trials(stimulus_presentations, stim)
        window = (natural_movies_window[0], stimulus_trials.duration + natural_movies_window[1])
        aligned_lfp, valid_blocks = st.align_trials_from_blocks(
            lfp_groups, st.get_stimulus_blocks(stimulus_trials), window=window, ignore_nan_trials='any')
        if not valid_blocks:
            print(f"Warning: All trials are dropped by NaN values in {stim}.")
            continue

        lfp_bands_power = []
        for wave_band in wave_bands:
            block_power = [bandpass_power(x) for x in ps.bandpass_filter_blocks(
                lfp_groups, valid_blocks,
                freq_bands.sel(wave_band=wave_band).values,
                extend_time=extend_time,
                include_filtered=False,
                include_amplitude=True,
                concat=False
            )]
            lfp_bands_power.append(st.align_trials_from_blocks(block_power, valid_blocks, window=window, ignore_nan_trials='')[0])
        lfp_bands_power = xr.concat(lfp_bands_power, dim=pd.Index(wave_bands, name='wave_band'), combine_attrs='drop_conflicts')

        if filter_instantaneous_power:
            block_filt = [x.filtered.assign_attrs(x.attrs) for x in ps.bandpass_filter_blocks(
                lfp_groups, valid_blocks, instantaneous_band, extend_time=extend_time, concat=False)]
            aligned_lfp = st.align_trials_from_blocks(block_filt, valid_blocks, window=window, ignore_nan_trials='')[0]
        lfp_power_dss[stim] = xr.Dataset(
            data_vars = dict(
                instantaneous_power = instantaneous_power(aligned_lfp),
                band_power = lfp_bands_power,
                freq_bands = freq_bands,
            )
        )


    #################### Save data ####################
    session_dir.save_stimulus_lfp_power(lfp_power_dss)


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(lfp_power_during_stimuli, **args)
