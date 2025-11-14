"""
Analyze PSD using FOOOF.

- Fit FOOOF to PSD of stimuli.
- Get frequency bands from FOOOF results.
- Calculate band power in drifting grating conditions.
- Save figures.
"""

import add_path

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
        default = 'beta',
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
    condition_wave_band: str = 'beta'
):
    import numpy as np
    import pandas as pd
    import xarray as xr
    import matplotlib.pyplot as plt

    import toolkit.allen_helpers.stimuli as st
    import toolkit.analysis.spectrum as spec
    import toolkit.plots.plots as plots
    import toolkit.plots.format as plt_fmt
    import toolkit.pipeline.location as pl
    from toolkit.pipeline.data_io import SessionDirectory, format_for_path
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS
    from toolkit.utils.quantity_units import as_string, as_quantity
    from toolkit.paths.paths import FIGURE_DIR


    #################### Get session and probe ####################
    session_dir = SessionDirectory(session_id, cache_lfp=True)

    probe_info = session_dir.load_probe_info()
    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    session = session_dir.session
    session_type = session.session_type

    drifting_gratings_stimuli = st.STIMULUS_CATEGORIES[session_type]['drifting_gratings']

    central_channels = probe_info['central_channels']
    layers = np.array(list(central_channels))[pl.argsort_channels(
        central_channels.values(), session_dir.load_lfp_channels())]

    #################### Load data and parameters ####################
    psd_das = session_dir.load_psd()
    cond_psd_das = session_dir.load_conditions_psd()

    stimulus_names = list(psd_das)
    stimulus_names.remove('channel_groups')

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
    fooof_objs = {}
    bands = np.full((len(stimulus_names), layers.size, wave_band_limit.wave_band.size, 2), np.nan)
    peaks = np.full(bands.shape[:-1] + (freq_band_kwargs['top_n_peaks'], ), np.nan)
    center_freq = peaks.copy()

    for i, stim in enumerate(stimulus_names):
        psd_avg = psd_das[stim]
        fooof_objs[stim] = {}
        for j, layer in enumerate(layers):
            psd_da = psd_avg.sel(layer=layer)

            # fit fooof
            fooof_result, fooof_objs[stim][layer], fooof_kwargs = spec.fit_fooof(psd_da, **fooof_params)
            gaussian_params = fooof_result.gaussian_params

            # get frequency bands
            for k, wave_band in enumerate(wave_band_limit.wave_band):
                band, peak_inds = spec.get_fooof_freq_band(
                    gaussian_params=gaussian_params,
                    freq_range=wave_band_limit.sel(wave_band=wave_band).values,
                    width_limit=wave_band_width_limit.sel(wave_band=wave_band).values,
                    **freq_band_kwargs
                )

                bands[i, j, k] = band
                peaks[i, j, k, :peak_inds.size] = gaussian_params[peak_inds, 1]
                center_freq[i, j, k, :peak_inds.size] = gaussian_params[peak_inds, 0]

    coords = dict(stimulus=stimulus_names, layer=layers, wave_band=wave_band_limit.coords['wave_band'])
    bands = xr.DataArray(data=bands, coords=coords | dict(bound=wave_band_limit.coords['bound']))
    peaks = xr.DataArray(data=peaks, coords=coords | dict(peak_rank=range(peaks.shape[-1])))
    center_freq = peaks.copy(data=center_freq)

    bands_ds = xr.Dataset(dict(
        bands=bands, peaks=peaks, center_freq=center_freq,
        wave_band_limit=wave_band_limit,
        wave_band_width_limit=wave_band_width_limit
    ))
    bands_ds.attrs.update(fooof_kwargs | fooof_params | freq_band_kwargs)

    # Get band power in drifting grating conditions
    fixed_condition_types = st.FIXED_CONDITION_TYPES[session_type]

    cond_band_power_das = {}

    for stim in drifting_gratings_stimuli:
        cond_band_power = {}
        for layer in layers:
            band = bands.sel(stimulus=stim, layer=layer, wave_band=condition_wave_band)
            if np.isnan(band).any():
                continue
            cond_psd = cond_psd_das[stim].sel(layer=layer)
            power = cond_psd.sel(frequency=slice(*band)).integrate('frequency').mean(dim=fixed_condition_types)
            unit = as_string(as_quantity(cond_psd.attrs['unit']) * as_quantity('Hz'))
            power.attrs.update(cond_psd.attrs | dict(unit=unit))
            cond_band_power[layer] = power

        if cond_band_power:
            cond_band_power_das[stim] = xr.concat(cond_band_power.values(),
                dim=pd.Index(cond_band_power, name='layer'), combine_attrs='override')
        else:
            print(f"No layer has detected {condition_wave_band} band during {stim}.")
            continue

    #################### Save data ####################
    session_dir.save_wave_bands(bands_ds)
    session_dir.save_condition_band_power(cond_band_power_das)

    #################### Save figures ####################
    if not plt_fmt.SAVE_FIGURE:
        return

    session_str = f"session_{session_id}"

    fig_dir = FIGURE_DIR / "PSD"
    layer_psd_dir = fig_dir / "layer_psd"
    fooof_dir = fig_dir / "fooof"
    cond_band_power_dir = fig_dir / f"condition_{condition_wave_band}_power"

    # Plot PSD and FOOOF
    for stim in stimulus_names:
        stim_layer_psd_dir = layer_psd_dir / stim
        stim_layer_psd_dir.mkdir(parents=True, exist_ok=True)
        stim_fooof_dir = fooof_dir / stim
        stim_fooof_dir.mkdir(parents=True, exist_ok=True)

        psd_avg = psd_das[stim]
        ax = plots.plot_channel_psd(psd_avg, channel_dim='layer', freq_range=plt_range)
        ax.set_title(stim)
        fig = ax.get_figure()
        plt_fmt.save_figure(stim_layer_psd_dir, fig, name=session_str)

        figs = {}
        for layer in layers:
            fig, ax = plt.subplots(1, 1)
            ax = plots.plot_fooof_quick(fooof_objs[stim][layer], freq_range=plt_range, ax=ax)

            band = bands.sel(stimulus=stim, layer=layer)
            ax = plots.plot_freq_band(band, band.wave_band, ax=ax)
            ax.set_title(f"{stim} layer {layer}")

            figs[f'{session_str}_layer_{format_for_path(layer)}'] = fig

        plt_fmt.save_figure(stim_fooof_dir, figs)
        plt.close('all')  # free memory

    # Plot band power in drifting grating conditions
    x_cond, y_cond = st.VARIED_CONDITION_TYPES[session_type]
    cond_label = st.COND_LABEL

    for stim, cond_band_power in cond_band_power_das.items():
        stim_cond_band_power_dir = cond_band_power_dir / stim
        stim_cond_band_power_dir.mkdir(parents=True, exist_ok=True)

        cond_power_db = 10 * np.log10(cond_band_power)
        vmin, vmax = cond_power_db.min(), cond_power_db.max()
        layers_ = cond_power_db.layer.values

        fig, axs = plt.subplots(layers_.size, 1, squeeze=False, figsize=(4.8, 3.0 * layers_.size))
        for layer, ax in zip(layers_, axs.ravel()):
            cpower = cond_power_db.sel(layer=layer).transpose(y_cond, x_cond)
            label = condition_wave_band.title() + ' power (dB)'
            band = bands.sel(stimulus=stim, layer=layer, wave_band=condition_wave_band).values

            x = cpower.coords[x_cond].values
            y = cpower.coords[y_cond].values
            cpower = cpower.assign_coords({x_cond: range(x.size), y_cond: range(y.size)})

            im = ax.imshow(cpower, origin='lower', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, label=label, pad=0.03)
            ax.set_xticks(cpower.coords[x_cond], labels=map('{:g}'.format, x))
            ax.set_yticks(cpower.coords[y_cond], labels=map('{:g}'.format, y))
            ax.set_xlabel(cond_label[x_cond])
            ax.set_ylabel(cond_label[y_cond])
            ax.set_title(f'Layer {layer}, {band[0]:.1f} - {band[1]:.1f} Hz')
        fig.tight_layout()

        plt_fmt.save_figure(stim_cond_band_power_dir, fig, name=session_str)

    plt.close('all')


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(analyze_psd_fooof, **args)
