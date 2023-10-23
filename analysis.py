import numpy as np
import xarray as xr
import scipy as sp
import matplotlib.pyplot as plt

import pywt
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic

"""
Functions for analyzing preprocessed data
"""

def fit_fooof(f, pxx, aperiodic_mode='fixed', dB_threshold=3., max_n_peaks=10,
              freq_range=None, peak_width_limits=None, report=False,
              plot=False, plt_log=False, plt_range=None, figsize=None):
    """Fit PSD using FOOOF"""
    if aperiodic_mode != 'knee':
        aperiodic_mode = 'fixed'
    def set_range(x, upper=f[-1]):        
        x = np.array(upper) if x is None else np.array(x)
        return [f[2], x.item()] if x.size == 1 else x.tolist()
    freq_range = set_range(freq_range)
    peak_width_limits = set_range(peak_width_limits, np.inf)

    # Initialize a FOOOF object
    fm = FOOOF(peak_width_limits=peak_width_limits, min_peak_height=dB_threshold / 10,
               peak_threshold=0., max_n_peaks=max_n_peaks, aperiodic_mode=aperiodic_mode)
    # Fit the model
    fm.fit(f, pxx, freq_range)
    results = fm.get_results()

    if report:
        fm.print_results()
        if aperiodic_mode=='knee':
            ap_params = results.aperiodic_params
            if ap_params[1] <= 0:
                print('Negative value of knee parameter occurred. Suggestion: Fit without knee parameter.')
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f'Knee location: {knee_freq:.2f} Hz')
    if plot:
        plt_range = set_range(plt_range)
        fm.plot(plt_log=plt_log)
        plt.xlim(np.log10(plt_range) if plt_log else plt_range)
        if figsize:
            plt.gcf().set_size_inches(figsize)
        ax = plt.gca()
        ax.legend(fontsize='medium', framealpha=0.2)
        ax.xaxis.label.set_size('medium')
        ax.yaxis.label.set_size('medium')
        ax.tick_params(labelsize='medium')
    return results, fm

def trial_psd(aligned_lfp, tseg=1.):
    """Calculate PSD from xarray of LFP using Welch method"""
    fs = aligned_lfp.fs
    trial_duration = aligned_lfp.time_from_presentation_onset[-1] - aligned_lfp.time_from_presentation_onset[0]
    nperseg = int(np.ceil(trial_duration / max(np.round(trial_duration / tseg), 1) * fs))
    f, pxx = sp.signal.welch(aligned_lfp.LFP, fs=fs, nperseg=nperseg)
    psd_array = xr.DataArray(pxx, coords={
        'channel': aligned_lfp.channel, 'presentation_id': aligned_lfp.presentation_id, 'frequency': f
    }).to_dataset(name='PSD')
    return psd_array

def plot_channel_psd(psd_avg, channel_id=None, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                 aperiodic_mode='knee', dB_threshold=3., max_n_peaks=10, plt_log=True):
    """Plot PSD at given chennel with FOOOF results"""
    psd_avg_plt = psd_avg.sel(frequency=slice(*plt_range))
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.get_cmap('plasma')(np.linspace(0, 1, psd_avg.dims['channel'])))
    plt.figure(figsize=figsize)
    plt.plot(psd_avg_plt.frequency, psd_avg_plt.PSD.T, label=psd_avg_plt.channel.values)
    plt.xlim(plt_range)
    plt.yscale('log')
    plt.legend(loc='upper right', framealpha=0.2, title='channel ID')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    fig1 = plt.gcf()

    if channel_id is None:
        channel_id = lfp_array.channel[0]
    print(f'Channel: {channel_id: d}')
    psd_avg_plt = psd_avg.sel(channel=channel_id)
    results = fit_fooof(psd_avg_plt.frequency.values, psd_avg_plt.PSD.values,
                        aperiodic_mode=aperiodic_mode, dB_threshold=dB_threshold, max_n_peaks=max_n_peaks,
                        freq_range=freq_range, peak_width_limits=None, report=True,
                        plot=True, plt_log=plt_log, plt_range=plt_range[1], figsize=figsize)
    fig2 = plt.gcf()
    return results, fig1, fig2

def cwt_spectrogram(x, fs, nNotes=6, nOctaves=np.inf, freq_range=(0, np.inf),
                    bandwidth=1.0, axis=-1, detrend=False, normalize=False):
    """Calculate spectrogram using continuous wavelet transform"""
    x = np.asarray(x)
    N = x.shape[axis]
    dt = 1 / fs
    times = np.arange(N) * dt
    # detrend and normalize
    if detrend:
        x = sp.signal.detrend(x, axis=axis, type='linear')
    if normalize:
        x = x / x.std()
    # Define some parameters of our wavelet analysis. 
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    nOctaves = min(nOctaves, np.log2(2 * np.floor(N / 2)))
    scales = 2 ** (1 + np.arange(np.floor(nOctaves * nNotes)) / nNotes)
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=2*bandwidth and center frequency of 1.0
    # bandwidth is the sigma of the gaussian envelope
    wavelet = 'cmor' + str(2 * bandwidth) + '-1.0'
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    scales = scales[(frequencies >= freq_range[0]) & (frequencies <= freq_range[1])]
    coef, frequencies = pywt.cwt(x, scales[::-1], wavelet=wavelet, sampling_period=dt, axis=axis)
    power = np.real(coef * np.conj(coef)) # equivalent to power = np.abs(coef)**2
    # smooth a bit
#     power = sp.ndimage.gaussian_filter(power, sigma=2, axes=(axis, ))
    # cone of influence in frequency for cmorxx-1.0 wavelet
    f0 = 2 * np.pi
    cmor_coi = 2 ** -0.5
    cmor_flambda = 4 * np.pi / (f0 + (2 + f0 ** 2) ** 0.5)
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    coi *= cmor_flambda * cmor_coi * dt
    # cone of influence in terms of frequency
    coif = 1 / coi
    return power, times, frequencies, coif

def trial_averaged_spectrogram(aligned_lfp, tseg=1., cwt=True, downsample_fs=200.):
    """Calculate average spectrogram over trials using Fourier or wavelet transform"""
    fs = aligned_lfp.fs
    if cwt:
        t = aligned_lfp.time_from_presentation_onset.values
        if downsample_fs is None:
            downsample_fs = fs
            downsampled = aligned_lfp.LFP.values
        else:
            num = int(t.size * downsample_fs / fs)
            downsample_fs = num / t.size * fs
            axis = aligned_lfp.LFP.dims.index('time_from_presentation_onset')
            downsampled, t = sp.signal.resample(aligned_lfp.LFP, num=num, t=t, axis=axis)
        sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, freq_range=(1 / tseg, np.inf), axis=axis)
        sxx = np.moveaxis(sxx, 0, -2)
    else:
        trial_duration = aligned_lfp.time_from_presentation_onset[-1] - aligned_lfp.time_from_presentation_onset[0]
        nperseg = int(np.ceil(trial_duration / max(np.round(trial_duration / tseg), 1) * fs))
        f, t, sxx = sp.signal.spectrogram(aligned_lfp.LFP.values, fs=fs, nperseg=nperseg)
        t += aligned_lfp.time_from_presentation_onset.values[0]
    sxx_avg = xr.DataArray(sxx, coords={
        'channel': aligned_lfp.channel, 'presentation_id': aligned_lfp.presentation_id, 'frequency': f, 'time': t
    }).mean(dim='presentation_id').to_dataset(name='PSD')
    if cwt:
        sxx_avg = sxx_avg.assign(cone_of_influence_frequency=xr.DataArray(coif, coords={'time': t}))
    return sxx_avg

def plot_spectrogram(sxx_xarray, remove_aperiodic=None, plt_log=False,
                     plt_range=None, clr_freq_range=None, ax=None):
    """Plot spectrogram. Determine color limits using value in frequency band clr_freq_range"""
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()
    f1_idx = 0 if f[0] else f[1]

    cbar_label = 'PSD' if remove_aperiodic is None else 'PSD Residual'
    if plt_log:
        with np.errstate(divide='ignore'):
            sxx = np.log10(sxx)
        cbar_label += ' log(power)'

    if remove_aperiodic is not None:
        ap_fit = gen_aperiodic(f[1:], remove_aperiodic.aperiodic_params)
        sxx[1:, :] -= (ap_fit if plt_log else 10 ** ap_fit)[:, None]
        sxx[0, :] = 0.

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [next(x for x in f if x) if plt_log else 0., plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    if clr_freq_range is None:
        vmin, vmax = None, None
    else:
        c_idx = (f >= clr_freq_range[0]) & (f <= clr_freq_range[1])
        vmin, vmax = sxx[c_idx, :].min(), sxx[c_idx, :].max()

    f = f[f_idx]
    pcm = ax.pcolormesh(t, f, sxx[f_idx, :], shading='gouraud', vmin=vmin, vmax=vmax)
    if 'cone_of_influence_frequency' in sxx_xarray:
        coif = sxx_xarray.cone_of_influence_frequency
        ax.plot(t, coif)
        ax.fill_between(t, coif, step='mid', alpha=0.2)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(f[0], f[-1])
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    return sxx

def plot_channel_spectrogram(sxx_avg, channel_id=None, plt_range=(0, 100.), plt_log=True,
                             clr_freq_range=None, figsize=(6, 3.6),
                             remove_aperiodic={'freq_range': 200., 'aperiodic_mode': 'knee'}):
    """Plot spectrograms of given channels"""
    if channel_id is None:
        channel_id = sxx_avg.channel.values
    else:
        channel_id = np.asarray(channel_id)
        if not channel_id.ndim:
            channel_id = channel_id.reshape(1)
    if remove_aperiodic is None:
        fooof_results = None

    _, axs = plt.subplots(channel_id.size, 1, squeeze=False, figsize=(figsize[0], channel_id.size * figsize[1]))
    for channel, ax in zip(channel_id, axs.ravel()):
        sxx_single = sxx_avg.sel(channel=channel)
        if remove_aperiodic is not None:
            sxx_tot = sxx_single.mean(dim='time')
            fooof_results, _ = fit_fooof(sxx_tot.frequency.values, sxx_tot.PSD.values, **remove_aperiodic)
        _ = plot_spectrogram(sxx_single, remove_aperiodic=fooof_results, plt_log=plt_log,
                             plt_range=plt_range, clr_freq_range=clr_freq_range, ax=ax)
        ax.set_title(f'channel {channel: d}')
    plt.tight_layout()
    return axs

def preprocess_firing_rate(units_fr, sigma, soft_normalize_cut, units_mean_fr=None):
    """Smooth and normalize units firing rate"""
    if units_mean_fr is None:
        units_mean_fr = units_fr.units_mean_fr
    axis = units_fr.spike_rate.dims.index('time_relative_to_stimulus_onset')
    smoothed = sp.ndimage.gaussian_filter1d(units_fr.spike_rate - units_mean_fr,
                                            sigma / units_fr.bin_width, axis=axis, mode='constant')
    smoothed = units_fr.spike_rate.copy(data=smoothed) + units_mean_fr
    units_fr = units_fr.assign(smoothed=smoothed)
    normalized = smoothed / (units_mean_fr + soft_normalize_cut)
    units_fr = units_fr.assign(normalized=normalized)
    return units_fr

def bandpass_lfp(aligned_lfp, filt_band, order=4, extend_time=None):
    """Filter LFP. Get amplitude and phase using Hilbert transform.
    extend_time: duration at the start and end to avoid boundary effect for filtering.
    """
    bfilt, afilt = sp.signal.butter(order, filt_band, btype='bandpass', fs=aligned_lfp.fs)
    axis = aligned_lfp.LFP.dims.index('time_from_presentation_onset')
    filtered = sp.signal.filtfilt(bfilt, afilt, aligned_lfp.LFP, axis=axis)
    analytic = sp.signal.hilbert(filtered)
    if extend_time is None:
        extend_time = aligned_lfp.extend_time
    filtered_lfp = xr.Dataset(
        data_vars = dict(
            LFP = aligned_lfp.LFP.copy(data=filtered),
            amplitude = aligned_lfp.LFP.copy(data=np.abs(analytic)),
            phase = aligned_lfp.LFP.copy(data=np.angle(analytic)),
        ),
        attrs = dict(
            fs = aligned_lfp.fs,
            filt_band = filt_band,
            extend_time=extend_time
        )
    )
    return filtered_lfp

def get_spike_phase(spike_times, filtered_lfp, unit_channel, time_window=None):
    """Get the LFP phase at each spike time. LFP was at the nearest channel to each unit."""
    unit_ids = unit_channel.index
    presentation_ids = filtered_lfp.presentation_id.to_index()
    if time_window is None:
        time_window = (filtered_lfp.time_from_presentation_onset.values[0] + filtered_lfp.extend_time,
                       filtered_lfp.time_from_presentation_onset.values[-1] - filtered_lfp.extend_time)
    spike_trains = [[[] for _ in range(presentation_ids.size)] for _ in range(unit_ids.size)]
    for row in spike_times.itertuples(index=False):
        i = unit_ids.get_loc(row.unit_id)
        j = presentation_ids.get_loc(row.stimulus_presentation_id)
        t = row.time_since_stimulus_presentation_onset
        if t >= time_window[0] and t <= time_window[1]:
            spike_trains[i][j].append(t)
    spike_trains = np.array(spike_trains, dtype=object)

    t0 = filtered_lfp.time_from_presentation_onset.values[0]
    fs = filtered_lfp.fs
    resultant_phase = np.zeros(spike_trains.shape, dtype=complex)
    spike_number = np.zeros(spike_trains.shape, dtype=int)
    for i in range(unit_ids.size):
        unit_phase = filtered_lfp.phase.sel(channel=unit_channel[unit_ids[i]])
        for j in range(presentation_ids.size):
            spike_train = np.array(spike_trains[i, j])
            spike_number[i, j] = spike_train.size
            t = np.round((spike_train - t0) * fs).astype(int)
            phase = unit_phase.isel(presentation_id=j, time_from_presentation_onset=t).values
            resultant_phase[i, j] = np.sum(np.exp(1j * phase))
    spike_phase = xr.Dataset(
        data_vars={
            'spike_number': (['unit_id', 'presentation_id'], spike_number),
            'resultant_phase': (['unit_id', 'presentation_id'], resultant_phase)
        },
        coords={'unit_id': unit_ids, 'presentation_id': presentation_ids},
        attrs={'time_window': time_window}
    )
    return spike_phase

def phase_locking_value(spike_phase, unit_ids=None, presentation_ids=None, unbiased=True):
    """Calculate phase locking value of each unit under given presentations"""
    if unit_ids is None:
        unit_ids = spike_phase.unit_id
    if presentation_ids is None:
        presentation_ids = spike_phase.presentation_id
    spk_pha = spike_phase.sel(unit_id=unit_ids, presentation_id=presentation_ids).sum(dim='presentation_id')
    pha = spk_pha.resultant_phase.values
    N = spk_pha.spike_number.values
    plv = np.zeros(pha.shape)
    if unbiased:
        plv_ub = np.zeros(pha.shape)
    idx = np.nonzero(N > 1)[0]
    if unbiased:
            plv2 = (pha[idx] * pha[idx].conj()).real / N[idx]
            plv[idx] = (plv2 / N[idx]) ** 0.5
            plv_ub[idx] = (np.fmax(plv2 - 1, 0) / (N[idx] - 1)) ** 0.5
    else:
        plv[idx] = np.abs(pha[idx]) / N[idx]
    mean_firing_rate = N / ((spike_phase.time_window[1] - spike_phase.time_window[0]) * len(presentation_ids))

    ds = xr.Dataset(
        data_vars={
            'PLV': (['unit_id'], plv),
            'phase': (['unit_id'], np.angle(pha, deg=True)),
            'mean_firing_rate': (['unit_id'], mean_firing_rate)
        },
        coords={'unit_id': unit_ids},
        attrs={'angle_unit': 'deg'}
    )
    if unbiased:
        ds = ds.assign({'PLV_unbiased': (['unit_id'], plv_ub)})
    return ds
