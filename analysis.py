import numpy as np
import xarray as xr
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pywt
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic, gen_model

"""
Functions for analyzing preprocessed data
"""

def fit_fooof(f, pxx, aperiodic_mode='fixed', dB_threshold=3., max_n_peaks=10,
              freq_range=None, peak_width_limits=None, report=False,
              plot=False, plt_log=False, plt_range=None, figsize=None, ax=None):
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
    try:
        fm.fit(f, pxx, freq_range)
    except Exception as e:
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        fm.fit(fl, np.interp(fl, f, pxx), freq_range)
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
        fm.plot(plt_log=plt_log, ax=ax)
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
    f, pxx = ss.welch(aligned_lfp.LFP, fs=fs, nperseg=nperseg)
    psd_array = xr.DataArray(pxx, coords={
        'channel': aligned_lfp.channel, 'presentation_id': aligned_lfp.presentation_id, 'frequency': f
    })
    return psd_array

def plot_channel_psd(psd_avg, channel_id=None, channel_coord='channel', channel_name='Channel ID',
                     freq_range=200., plt_range=(0, 100.), figsize=(5, 4), ax1=None, ax2=None,
                     aperiodic_mode='knee', dB_threshold=3., max_n_peaks=10, plot=True, plt_log=True):
    """Plot PSD at given chennel with FOOOF results"""
    plt_range = np.array(plt_range)
    if plt_range.size == 1:
        plt_range = (0, plt_range.item())
    psd_avg_plt = psd_avg.sel(frequency=slice(*plt_range))

    if ax1 is None:
        _, ax1 = plt.subplots(1, 1, figsize=figsize)
    ax = ax1
    fig1 = ax.get_figure()
    ax.set_prop_cycle(plt.cycler('color', plt.cm.get_cmap('plasma')(
        np.linspace(0, 1, psd_avg.coords[channel_coord].size))))
    ax.plot(psd_avg_plt.frequency, psd_avg_plt.values.T, label=psd_avg_plt.coords[channel_coord].values)
    ax.set_xlim(plt_range)
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.2, title=channel_name)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')

    if channel_id is None:
        return None, fig1, None
    print(f'{channel_name:s}: {channel_id}')
    psd_avg_plt = psd_avg.sel({channel_coord: channel_id})
    if ax2 is None:
        _, ax2 = plt.subplots(1, 1, figsize=figsize)
    fig2 = ax2.get_figure()
    results = fit_fooof(psd_avg_plt.frequency.values, psd_avg_plt.values,
                        aperiodic_mode=aperiodic_mode, dB_threshold=dB_threshold, max_n_peaks=max_n_peaks,
                        freq_range=freq_range, peak_width_limits=None, report=True,
                        plot=plot, plt_log=plt_log, plt_range=plt_range[1], figsize=figsize, ax=ax2)
    return results, fig1, fig2

def plot_fooof(f, pxx, fooof_result, plt_log=False, plt_range=None, plt_db=True, ax=None):
    full_fit, _, ap_fit = gen_model(f[1:], fooof_result.aperiodic_params,
                                    fooof_result.gaussian_params, return_components=True)
    full_fit = np.insert(10 ** full_fit, 0, pxx[0])
    ap_fit = np.insert(10 ** ap_fit, 0, pxx[0])

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[1] if plt_log else 0., plt_range.item()]
    f_idx = (f >= plt_range[0]) & (f <= plt_range[1])
    f, pxx = f[f_idx], pxx[f_idx]
    full_fit, ap_fit = full_fit[f_idx], ap_fit[f_idx]
    if plt_db:
        pxx, full_fit, ap_fit = [10 * np.log10(x) for x in [pxx, full_fit, ap_fit]]
    ax.plot(f, pxx, 'k', label='Original')
    ax.plot(f, full_fit, 'r', label='Full model fit')
    ax.plot(f, ap_fit, 'b--', label='Aperiodic fit')
    if plt_log:
        ax.set_xscale('log')
    ax.set_xlim(plt_range)
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD ' + ('dB' if plt_db else r'mV$^2$') + '/Hz')
    return fig, ax

# cone of influence in frequency for cmorxx-1.0 wavelet
f0 = 2 * np.pi
CMOR_COI = 2 ** -0.5
CMOR_FLAMBDA = 4 * np.pi / (f0 + (2 + f0 ** 2) ** 0.5)
COI_FREQ = 1 / (CMOR_COI * CMOR_FLAMBDA)

def cwt_spectrogram(x, fs, nNotes=6, nOctaves=np.inf, freq_range=(0, np.inf),
                    bandwidth=1.0, axis=-1, detrend=False, normalize=False):
    """Calculate spectrogram using continuous wavelet transform"""
    x = np.asarray(x)
    N = x.shape[axis]
    times = np.arange(N) / fs
    # detrend and normalize
    if detrend:
        x = ss.detrend(x, axis=axis, type='linear')
    if normalize:
        x = x / x.std()
    # Define some parameters of our wavelet analysis. 
    # range of scales (in time) that makes sense
    # min = 2 (Nyquist frequency)
    # max = np.floor(N/2)
    nOctaves = min(nOctaves, np.log2(2 * np.floor(N / 2)))
    scales = 2 ** np.arange(1, nOctaves, 1 / nNotes)
    # cwt and the frequencies used. 
    # Use the complex morelet with bw=2*bandwidth^2 and center frequency of 1.0
    # bandwidth is sigma of the gaussian envelope
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    frequencies = pywt.scale2frequency(wavelet, scales) * fs
    scales = scales[(frequencies >= freq_range[0]) & (frequencies <= freq_range[1])]
    coef, frequencies = pywt.cwt(x, scales[::-1], wavelet=wavelet, sampling_period=1 / fs, axis=axis)
    power = np.real(coef * np.conj(coef)) # equivalent to power = np.abs(coef)**2
    # cone of influence in terms of wavelength
    coi = N / 2 - np.abs(np.arange(N) - (N - 1) / 2)
    # cone of influence in terms of frequency
    coif = COI_FREQ * fs / coi
    return power, times, frequencies, coif

def trial_averaged_spectrogram(aligned_lfp, tseg=1., cwt=True, downsample_fs=200.):
    """Calculate average spectrogram over trials using Fourier or wavelet transform"""
    fs = aligned_lfp.fs
    axis = aligned_lfp.LFP.dims.index('time_from_presentation_onset')
    if cwt:
        t = aligned_lfp.time_from_presentation_onset.values
        if downsample_fs is None or downsample_fs >= fs:
            downsample_fs = fs
            downsampled = aligned_lfp.LFP.values
        else:
            num = int(t.size * downsample_fs / fs)
            downsample_fs = num / t.size * fs
            downsampled, t = ss.resample(aligned_lfp.LFP.values, num=num, t=t, axis=axis)
        downsampled = np.moveaxis(downsampled, axis, -1)
        sxx, _, f, coif = cwt_spectrogram(downsampled, downsample_fs, freq_range=(1 / tseg, np.inf), axis=-1)
        sxx = np.moveaxis(sxx, 0, -2) # shape (... , freq, time)
    else:
        trial_duration = aligned_lfp.time_from_presentation_onset[-1] - aligned_lfp.time_from_presentation_onset[0]
        nperseg = int(np.ceil(trial_duration / max(np.round(trial_duration / tseg), 1) * fs))
        f, t, sxx = ss.spectrogram(np.moveaxis(aligned_lfp.LFP.values, axis, -1), fs=fs, nperseg=nperseg)
        t += aligned_lfp.time_from_presentation_onset.values[0]
    sxx_avg = xr.DataArray(sxx, coords={
        'channel': aligned_lfp.channel, 'presentation_id': aligned_lfp.presentation_id, 'frequency': f, 'time': t
    }).mean(dim='presentation_id').to_dataset(name='PSD')
    if cwt:
        sxx_avg = sxx_avg.assign(cone_of_influence_frequency=xr.DataArray(coif, coords={'time': t}))
    return sxx_avg

def plot_spectrogram(sxx_xarray, remove_aperiodic=None, log_power=False,
                     plt_range=None, clr_freq_range=None, pad=0.03, ax=None):
    """Plot spectrogram. Determine color limits using value in frequency band clr_freq_range"""
    sxx = sxx_xarray.PSD.values.copy()
    t = sxx_xarray.time.values.copy()
    f = sxx_xarray.frequency.values.copy()

    cbar_label = 'PSD' if remove_aperiodic is None else 'PSD Residual'
    if log_power:
        with np.errstate(divide='ignore'):
            sxx = np.log10(sxx)
        cbar_label += ' dB' if log_power == 'dB' else ' log(power)'

    if remove_aperiodic is not None:
        f1_idx = 0 if f[0] else 1
        ap_fit = gen_aperiodic(f[f1_idx:], remove_aperiodic.aperiodic_params)
        sxx[f1_idx:, :] -= (ap_fit if log_power else 10 ** ap_fit)[:, None]
        sxx[:f1_idx, :] = 0.

    if log_power == 'dB':
        sxx *= 10

    if ax is None:
        _, ax = plt.subplots(1, 1)
    plt_range = np.array(f[-1]) if plt_range is None else np.array(plt_range)
    if plt_range.size == 1:
        plt_range = [f[0 if f[0] else 1] if log_power else 0., plt_range.item()]
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
    plt.colorbar(mappable=pcm, ax=ax, label=cbar_label, pad=pad)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    return sxx

def plot_channel_spectrogram(sxx_avg, channel_id=None, plt_range=(0, 100.), log_power=True,
                             clr_freq_range=None, pad=0.03, figsize=(6, 3.6),
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
            sxx_tot = sxx_single.PSD.mean(dim='time')
            fooof_results, _ = fit_fooof(sxx_tot.frequency.values, sxx_tot.values, **remove_aperiodic)
        _ = plot_spectrogram(sxx_single, remove_aperiodic=fooof_results, log_power=log_power,
                             plt_range=plt_range, clr_freq_range=clr_freq_range, pad=pad, ax=ax)
        ax.set_title(f'channel {channel: d}')
    plt.tight_layout()
    return axs

def bandpass_lfp(aligned_lfp, filt_band, order=4, extend_time=None, output='ba', include_analytic=False):
    """Filter LFP. Get amplitude and phase using Hilbert transform.
    extend_time: duration at the start and end to avoid boundary effect for filtering.
    """
    filt = ss.butter(order, filt_band, btype='bandpass', fs=aligned_lfp.fs, output=output)
    axis = aligned_lfp.LFP.dims.index('time_from_presentation_onset')
    if output == 'ba':
        filtered = ss.filtfilt(*filt, aligned_lfp.LFP, axis=axis)
    elif output == 'sos':
        filtered = ss.sosfiltfilt(filt, aligned_lfp.LFP, axis=axis)
    else:
        raise ValueError(f"Filter type {output} not supported.")
    analytic = ss.hilbert(filtered)
    if extend_time is None:
        extend_time = aligned_lfp.extend_time
    filtered_lfp = xr.Dataset(
        data_vars = dict(
            LFP = aligned_lfp.LFP.copy(data=filtered),
            amplitude = aligned_lfp.LFP.copy(data=np.abs(analytic)),
            phase = aligned_lfp.LFP.copy(data=np.angle(analytic))
        ),
        attrs = dict(
            fs = aligned_lfp.fs,
            filt_band = filt_band,
            extend_time=extend_time
        )
    )
    if include_analytic:
        filtered_lfp = filtered_lfp.assign(analytic = aligned_lfp.LFP.copy(data=analytic))
    return filtered_lfp

def get_spike_phase(spike_times, filtered_lfp, unit_channel, time_window=None):
    """Get the LFP phase at each spike time.
    unit_channel: specified channel to each unit at which LFP was used
    time_window: time window in which spikes are considered
    """
    unit_channel = xr.DataArray(unit_channel)
    unit_ids = unit_channel.get_index('unit_id')
    presentation_ids = filtered_lfp.presentation_id.to_index()
    if time_window is None:
        time_window = (filtered_lfp.time_from_presentation_onset.values[0] + filtered_lfp.extend_time,
                       filtered_lfp.time_from_presentation_onset.values[-1] - filtered_lfp.extend_time)
    spike_trains = np.full((unit_ids.size, presentation_ids.size), None, dtype=object)
    spike_trains.ravel()[:] = [[] for _ in range(spike_trains.size)]
    for row in spike_times.itertuples(index=False):
        i = unit_ids.get_loc(row.unit_id)
        j = presentation_ids.get_loc(row.stimulus_presentation_id)
        t = row.time_since_stimulus_presentation_onset
        if t >= time_window[0] and t <= time_window[1]:
            spike_trains[i][j].append(t)
    spike_trains = np.array(spike_trains, dtype=object)

    interpolate = 'analytic' in filtered_lfp  # automatically choose method
    if not interpolate:
        t0 = filtered_lfp.time_from_presentation_onset.values[0]
        fs = filtered_lfp.fs
    channel_coords = {k: v for k, v in unit_channel.coords.items() if k != 'unit_id'}
    channel_dims = [k for k in unit_channel.dims if k in channel_coords]
    channel_shape = tuple(unit_channel.coords[k].size for k in channel_dims)
    resultant_phase = np.zeros(spike_trains.shape + channel_shape, dtype=complex)
    spike_number = np.zeros(spike_trains.shape, dtype=int)
    for i in range(unit_ids.size):
        channel = unit_channel.sel(unit_id=unit_ids[i]).values
        if interpolate:
            unit_analytic = filtered_lfp.analytic.sel(channel=channel)
        else:
            unit_phase = filtered_lfp.phase.sel(channel=channel)
        for j in range(presentation_ids.size):
            spike_train = np.array(spike_trains[i, j])
            spike_number[i, j] = spike_train.size
            if not spike_train.size:  # no spikes, skip
                continue
            if interpolate:  # linear interpolate analytic signal
                analytic = unit_analytic.isel(presentation_id=j).interp(
                    time_from_presentation_onset=spike_train, assume_sorted=True)
                analytic = analytic / np.abs(analytic)  # normalize to unit circle
            else:  # nearest neighbor of phase
                t = np.round((spike_train - t0) * fs).astype(int)
                phase = unit_phase.isel(presentation_id=j, time_from_presentation_onset=t)
                analytic = np.exp(1j * phase)
            resultant_phase[i, j] = analytic.sum(dim='time_from_presentation_onset').values
    spike_phase = xr.Dataset(
        data_vars={
            'spike_number': (['unit_id', 'presentation_id'], spike_number),
            'resultant_phase': (['unit_id', 'presentation_id', *channel_dims], resultant_phase)
        },
        coords={'unit_id': unit_ids, 'presentation_id': presentation_ids, **channel_coords},
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
    pha = spk_pha.resultant_phase
    dims, coords = pha.dims, pha.coords
    pha = pha.values
    N = spk_pha.spike_number.values
    plv = np.zeros(pha.shape)
    if unbiased:
        plv_ub = np.zeros(pha.shape)
        ppc = np.zeros(pha.shape)
    idx = np.nonzero(N > 1)
    if unbiased:
            plv2 = (pha[idx] * pha[idx].conj()).real / N[idx]
            plv[idx] = (plv2 / N[idx]) ** 0.5
            ppc[idx] = np.fmax(plv2 - 1, 0) / (N[idx] - 1)
            plv_ub[idx] = ppc[idx] ** 0.5
    else:
        plv[idx] = np.abs(pha[idx]) / N[idx]
    mean_firing_rate = N / ((spike_phase.time_window[1] - spike_phase.time_window[0]) * len(presentation_ids))

    ds = xr.Dataset(
        data_vars={
            'PLV': (dims, plv),
            'phase': (dims, np.angle(pha, deg=True)),
            'mean_firing_rate': (dims, mean_firing_rate)
        },
        coords=coords, attrs={'angle_unit': 'deg'}
    )
    if unbiased:
        ds = ds.assign({'PLV_unbiased': (dims, plv_ub), 'PPC': (dims, ppc)})
    return ds

def wave_hilbert(x, freq_band, fs, filt_order=4, axis=-1):
    sos = ss.butter(N=filt_order, Wn=freq_band, btype='bandpass', fs=fs, output='sos')
    x_a = ss.hilbert(ss.sosfiltfilt(sos, np.asarray(x), axis=axis), axis=axis)
    return x_a

def wave_cwt(x, freq, fs, bandwidth=1.0, axis=-1):
    wavelet = 'cmor' + str(2 * bandwidth ** 2) + '-1.0'
    x_a = pywt.cwt(np.asarray(x), fs / freq, wavelet=wavelet, axis=axis)[0][0]
    return x_a

def get_waves(x, fs, waves, transform, axis=-1, component='amp', **kwargs):
    x = np.asarray(x)
    y = np.empty((len(waves),) + x.shape)
    comp_funcs = {'amp': np.abs, 'pha': np.angle}
    comp_func = comp_funcs.get(component, comp_funcs['amp'])
    for i, freq in enumerate(waves.values()):
        y[i][:] = comp_func(transform(x, freq, fs, axis=axis, **kwargs))
    return y

def spike_count(tspk, bin_edges):
    """Count spikes given in list of spike times into bins"""
    ispk = np.digitize(tspk, bin_edges)
    cspk = np.zeros(bin_edges.size + 1, dtype=int)
    for i in ispk:
        cspk[i] += 1
    return cspk[1:-1]

def gauss_filt_da(da, filt_sigma, dim='time'):
    """Gaussian smooth xr.DataArray along specific dimension"""
    dims = da.dims
    axis = dims.index(dim)
    sigma = [0] * len(dims)
    sigma[axis] = filt_sigma
    filt = gaussian_filter(da, sigma)
    return xr.DataArray(filt, coords=da.coords, dims=dims)

def gauss_filt(x, filt_sigma, axis=-1):
    """Gaussian smooth numpy array along specific dimension"""
    x = np.asarray(x).astype(float)
    sigma = [0] * x.ndim
    sigma[axis] = filt_sigma
    return gaussian_filter(x, sigma)

def exponential_spike_filter(spikes, tau, cut_val=1e-3, min_rate=None,
                             normalize=False, last_jump=True, only_jump=False):
    """Filter spike train (boolean/int array) with exponential response
    spikes: spike count array (time bins along the last axis)
    tau: time constant of the exponential decay (normalized by time step)
    cut_val: value at which to cutoff the tail of the exponential response
    min_rate: minimum rate of spike (normalized by sampling rate). Default: 1/(9*tau)
        It ensures the filtered values not less than min_val=exp(-1/(min_rate*tau)).
        It also ensures the jump value not less than 1+min_val.
        Specify min_rate=0 to set min_val to 0.
    normalize: whether normalize response to have integral 1 for filtering
    last_jump: whether return a time series with value at each time point equal
        to the unnormalized filtered value at the last spike (jump value)
    only_jump: whether return jump values only at spike times, 0 at non-spike time
    """
    spikes = np.asarray(spikes).astype(float)
    shape = spikes.shape
    if tau <= 0:
        filtered = spikes
        if only_jump:
            jump = spikes.copy()
        elif last_jump:
            jump = np.ones(shape)
    else:
        spikes = spikes.reshape(-1, shape[-1])
        min_val = np.exp(-9) if min_rate is None else \
            (0 if min_rate <= 0 else np.exp(-1 / min_rate / tau))
        t_cut = int(np.ceil(-np.log(cut_val) * tau))
        response = np.exp(-np.arange(t_cut) / tau)[None, :]
        filtered = ss.convolve(spikes, response, mode='full')
        filtered = np.fmax(filtered[:, :shape[-1]], min_val)
        if only_jump:
            idx = spikes > 0
            jump = np.where(idx, filtered, 0)
            if min_val > 0:
                jump[idx] = np.fmax(jump[idx], 1 + min_val)
        elif last_jump:
            min_val = 1 + min_val
            jump = filtered.copy()
            for jp, spks in zip(jump, spikes):
                idx = np.nonzero(spks)[0].tolist() + [None]
                jp[None:idx[0]] = min_val
                for i in range(len(idx) - 1):
                    jp[idx[i]:idx[i + 1]] = max(jp[idx[i]], min_val)
        if normalize:
            filtered /= np.sum(response)
        filtered = filtered.reshape(shape)
    if last_jump or only_jump:
        jump = jump.reshape(shape)
        filtered = (filtered, jump)
    return filtered

def stp_weights(rspk, tau, unit_axis=-2, time_axis=-1, i_start=0, i_stop=None):
    """Compute STP weigths from preprocessed spike rate
    tau: exponential filter time constant (time steps), scalar or array
    """
    tau = np.asarray(tau)
    rspk = np.asarray(rspk)
    slc = [slice(None)] * rspk.ndim
    slc[time_axis] = slice(i_start, i_stop)
    slc = tuple(slc)
    rspk_exp_filt = [np.moveaxis(exponential_spike_filter(np.moveaxis(rspk, time_axis, -1),
        tau=t, min_rate=0, normalize=True, last_jump=False), -1, time_axis)[slc] for t in tau.ravel()]
    rspk_exp_filt = np.stack(rspk_exp_filt, axis=0)
    rspk = rspk[slc]
    w_stp = rspk * rspk_exp_filt
    fr_tot = rspk
    if unit_axis is not None:
        w_stp = np.mean(w_stp, axis=unit_axis)
        fr_tot = np.mean(fr_tot, axis=unit_axis)
    return w_stp, fr_tot

def quantize(x, n_bins):
    """Quantize data in an array by its equally spaced quantiles
    x: data array
    n_bins: number of quantile bins of equal space
    Return: array of bin index of data points, value of bin edges
    """
    x = np.asarray(x)
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bid = np.digitize(x, bins[1:-1])
    return bid, bins

def statistic_in_quantile_grid(X, Y, n_bins=8, stat=np.mean, stat_fill=np.nan):
    """Divide data points into grids by n-quantiles of some features and
    calculate statistics of some features in the grids
    X: list of arrays of features according to which data are divided
    Y: list of arrays of features of which to obtain statistics
    n_bins: number of bins for all features in X
        or a list of them corresponding to each feature in X
    stat: function that calculate a statistic of each feature in Y. default: mean
        function should allow operation along specific axis with argument `axis`
    stat_fill: value to fill when no data exists in a grid. default: nan
    Return: statistics of each feature in Y, bin edges of each features in X, nd histogram count
    Note: data in X and Y must not contain nan values
    """
    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(X)
    else:
        if len(X) != len(n_bins):
            raise ValueError("Size of `n_bins` should match number of features `X`")
    bids, bins = zip(*map(quantize, X, n_bins))
    bidx = [np.arange(n) for n in n_bins]
    gids = [bid == idx[:, None] for bid, idx in zip(bids, bidx)]
    grid_ids = np.meshgrid(*bidx, indexing='ij')
    idx_in_grid = np.full(n_bins, None, dtype=object)
    hist_count = np.zeros(n_bins, dtype=int)
    for ids in np.nditer(grid_ids):
        idx = np.all([gid[i] for gid, i in zip(gids, ids)], axis=0)
        idx_in_grid[ids] = np.nonzero(idx)[0]
        hist_count[ids] = idx_in_grid[ids].size
    y_stats = []
    for y in Y:
        y = np.asarray(y)
        y_stat = np.full(n_bins, stat_fill, dtype=y.dtype)
        for ids in np.nditer(grid_ids):
            if hist_count[ids]:
                y_stat[ids] = stat(y[idx_in_grid[ids]])
        y_stats.append(y_stat)
    y_stats = np.stack(y_stats, axis=0)
    return y_stats, bins, hist_count

def detect_optotag(optotag_df, evoked_ratio_threshold=1.5, ttest_alpha=0.05, spike_width_threshold=0.5):
    """Detect optotag units using evoked ratio and spike width threshold from dataframe"""
    if evoked_ratio_threshold is None:
        optotag_df['evoke_positive'] = True
    else:
        optotag_df['evoke_positive'] = optotag_df['evoked_ratio'] > evoked_ratio_threshold
    if ttest_alpha is None:
        optotag_df['evoke_significant'] = True
    else:
        optotag_df['evoke_significant'] = ~optotag_df['p_value'].isna() & (optotag_df['p_value'] < ttest_alpha)
    if spike_width_threshold is None:
        optotag_df['low_spike_width'] = True
    else:
        optotag_df['low_spike_width'] = optotag_df['waveform_duration'] <= spike_width_threshold
    optotag_df['positive'] = optotag_df['evoke_positive'] & optotag_df['evoke_significant'] & optotag_df['low_spike_width']
    positive_units = optotag_df.index[optotag_df['positive']].values
    return optotag_df, positive_units
