"""
Analyze PSD using FOOOF.

- Fit FOOOF to PSD of stimuli.
- Get frequency bands from FOOOF results.
- Calculate band power in drifting grating conditions.
- Save figures.
"""

import add_path
from toolkit.pipeline.global_settings import GLOBAL_SETTINGS

PARAMETERS = dict(
    freq_range = dict(
        default = [200.],
        type = list,
        help = "Frequency range to fit for FOOOF. [min, max] Hz or [max] Hz."
    ),
    aperiodic_mode = dict(
        default = 'knee',
        type = str,
        help = "Aperiodic mode for FOOOF. 'knee' or 'fixed'."
    ),
    peak_width_limits = dict(
        default = [0., float('inf')],
        type = list,
        help = "Peak width limits for FOOOF. [min, max] Hz."
    ),
    max_n_peaks = dict(
        default = 10,
        type = int,
        help = "Maximum number of peaks for FOOOF."
    ),
    dB_threshold = dict(
        default = 0.8,
        type = float,
        help = "dB threshold for `min_peak_height` of FOOOF."
    ),
    peak_threshold = dict(
        default = 1.0,
        type = float,
        help = "Peak threshold for FOOOF in terms of the standard deviation of the aperiodic-removed power spectrum."
    ),
    plt_range = dict(
        default = 100.,
        type = float,
        help = "Frequency range for plotting PSD."
    ),
    top_n_peaks = dict(
        default = 2,
        type = int,
        help = "Number of top peaks to include to get the frequency band from FOOOF results."
    ),
    bandwidth_n_sigma = dict(
        default = 1.5,
        type = float,
        help = "Multiplier of sigma of the Gaussian parameters to define the bandwidth of the peak."
    ),
    condition_wave_band = dict(
        default = GLOBAL_SETTINGS['condition_wave_band'],
        type = str,
        help = "Wave band to analyze in drifting grating conditions."
    )
)


def analyze_psd_fooof(
    session_id: int,
    freq_range: list[float],
    aperiodic_mode: str,
    peak_width_limits: list[float],
    max_n_peaks: int,
    dB_threshold: float,
    peak_threshold: float,
    plt_range: float,
    top_n_peaks: int = 2,
    bandwidth_n_sigma: float = 1.5,
    condition_wave_band: str = GLOBAL_SETTINGS['condition_wave_band']
):
    import xarray as xr
    import matplotlib.pyplot as plt

    import toolkit.allen_helpers.stimuli as st
    import toolkit.plots.plots as plots
    import toolkit.pipeline.signal as ps
    from toolkit.pipeline.data_io import SessionDirectory, format_for_path
    from toolkit.paths.paths import FIGURE_DIR
    from toolkit.plots.format import SAVE_FIGURE, save_figure


    #################### Get session and probe ####################
    session_dir = SessionDirectory(session_id, cache_lfp=True)

    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    session = session_dir.session
    session_type = session.session_type

    drifting_gratings_stimuli = st.STIMULUS_CATEGORIES[session_type]['drifting_gratings']

    #################### Load data and parameters ####################
    psd_ds, channel_groups = session_dir.load_psd()
    cond_psd_das = session_dir.load_conditions_psd()

    fooof_params = dict(
        freq_range=freq_range,
        aperiodic_mode=aperiodic_mode,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        dB_threshold=dB_threshold,
        peak_threshold=peak_threshold,
    )

    freq_band_kwargs = dict(top_n_peaks=top_n_peaks, bandwidth_n_sigma=bandwidth_n_sigma)

    wave_band_limit = GLOBAL_SETTINGS['wave_band_limit']
    wave_band_limit = xr.DataArray(list(wave_band_limit.values()),
        coords=dict(wave_band=list(wave_band_limit), bound=['lower', 'upper']))
    wave_band_width_limit = wave_band_limit.copy(
        data=list(GLOBAL_SETTINGS['wave_band_width_limit'].values()))

    #################### Analyze data ####################
    # Fit FOOOF and get frequency bands
    fooof_objs, bands_ds = ps.fit_fooof_and_get_bands(
        psd_ds, fooof_params, freq_band_kwargs, wave_band_limit, wave_band_width_limit)

    # Get band power in drifting grating conditions
    fixed_condition_types = st.FIXED_CONDITION_TYPES[session_type]

    cond_band_power_das = {}
    layer_bands_ds = {}
    for stim in drifting_gratings_stimuli:
        cond_band_power_das[stim], layer_bands_ds[stim] = ps.layer_condition_band_power(
            cond_psd_das[stim].sel(stimulus=stim), bands_ds.bands.sel(stimulus=stim),
            wave_band_limit, fixed_condition_types, condition_wave_band=condition_wave_band
        )


    # Drifting grating PSD with filtered conditions
    layer_of_interest = GLOBAL_SETTINGS['layer_of_interest']
    drifting_gratings_windows = GLOBAL_SETTINGS['drifting_gratings_windows']

    # filter drifting gratings conditions
    cond_band_power = cond_band_power_das[drifting_gratings_stimuli[0]]  # first drifting grating stimulus as primary
    cond_band_power = ps.filter_conditions(cond_band_power)  # default filters defined in GLOBAL_SETTINGS['condition_filters']
    # find preferred orientation with max power
    average_dims = tuple(d for d in cond_band_power.dims if d in st.CONDITION_TYPES and d != 'orientation')
    preferred_orientations = cond_band_power.mean(dim=average_dims).idxmax(dim='orientation')

    # get PSD with filtered conditions
    filt_cond_psd = {}
    preferred_orientation = preferred_orientations.sel(layer=[layer_of_interest]).values
    for stim in drifting_gratings_stimuli:
        cond_psd = ps.filter_conditions(cond_psd_das[stim]).sel(orientation=preferred_orientation)
        cond_psd = cond_psd.mean(dim=st.CONDITION_TYPES)
        filt_cond_psd[stim + '_filtered'] = psd_ds[stim].copy(data=cond_psd.sel(stimulus=stim))
        for window_name in drifting_gratings_windows:
            stim_name = f'{stim}_{window_name}'
            filt_cond_psd[stim_name + '_filtered'] = psd_ds[stim].copy(data=cond_psd.sel(stimulus=stim_name))

    # update PSD of stimuli
    psd_ds = psd_ds.assign(filt_cond_psd)

    # fit FOOOF and get frequency bands
    filt_fooof_objs, filt_bands_ds = ps.fit_fooof_and_get_bands(
        filt_cond_psd, fooof_params, freq_band_kwargs, wave_band_limit, wave_band_width_limit)

    # update FOOOF objects
    fooof_objs.update(filt_fooof_objs)
    # add bands of filtered stimuli to bands_ds if not exist
    add_stimulus = [s for s in filt_bands_ds.coords['stimulus'].values if s not in bands_ds.coords['stimulus']]
    bands_ds = xr.concat([bands_ds, filt_bands_ds.sel(stimulus=add_stimulus)],
        data_vars=['bands', 'peaks', 'center_freq'], dim='stimulus')


    #################### Save data ####################
    # Save band power in drifting grating conditions
    session_dir.save_condition_band_power(cond_band_power_das, wave_band=condition_wave_band)

    # Save preferred orientations
    session_dir.save_preferred_orientations(preferred_orientations, wave_band=condition_wave_band)

    # Save updated PSD of stimuli
    session_dir.save_psd(psd_ds, channel_groups)

    # Save bands
    session_dir.save_wave_bands(bands_ds)


    #################### Save figures ####################
    if not SAVE_FIGURE:
        return

    session_str = f"session_{session_id}"

    fig_dir = FIGURE_DIR / "PSD"
    layer_psd_dir = fig_dir / "layer_psd"
    fooof_dir = fig_dir / "fooof"
    cond_band_power_dir = fig_dir / f"condition_{condition_wave_band}_power"

    # Plot PSD and FOOOF
    for stim, psd_avg in psd_ds.data_vars.items():
        stim_layer_psd_dir = layer_psd_dir / stim
        stim_layer_psd_dir.mkdir(parents=True, exist_ok=True)
        stim_fooof_dir = fooof_dir / stim
        stim_fooof_dir.mkdir(parents=True, exist_ok=True)

        ax = plots.plot_channel_psd(psd_avg, channel_dim='layer', freq_range=plt_range)
        ax.set_title(stim)
        fig = ax.get_figure()
        save_figure(stim_layer_psd_dir, fig, name=session_str)

        figs = {}
        for layer in psd_avg.coords['layer'].values:
            fig, ax = plt.subplots(1, 1)
            ax = plots.plot_fooof_quick(fooof_objs[stim][layer], freq_range=plt_range, ax=ax)

            band = bands_ds.bands.sel(stimulus=stim, layer=layer)
            ax = plots.plot_freq_band(band, band.wave_band, ax=ax)
            ax.set_title(f"{stim} layer {layer}")

            figs[f'{session_str}_layer_{format_for_path(layer)}'] = fig

        save_figure(stim_fooof_dir, figs)
        plt.close('all')  # free memory

    # Plot band power in drifting grating conditions
    x_cond, y_cond = st.VARIED_CONDITION_TYPES[session_type]
    for stim, cond_band_power in cond_band_power_das.items():
        stim_cond_band_power_dir = cond_band_power_dir / stim
        stim_cond_band_power_dir.mkdir(parents=True, exist_ok=True)

        fig, axs = plots.plot_layer_condition_band_power(cond_band_power, layer_bands_ds[stim], x_cond, y_cond)
        save_figure(stim_cond_band_power_dir, fig, name=session_str)

    plt.close('all')


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(analyze_psd_fooof, **args)
