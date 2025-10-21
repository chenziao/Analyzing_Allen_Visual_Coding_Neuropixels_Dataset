from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
import scipy.signal as ss
import warnings

import pywt
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic, gen_model

from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from fooof.data import FOOOFResults


def _get_psd_freq_range(
    freq_range : ArrayLike | float,
    frequencies : ArrayLike | None = None,
    log_frequency : bool = False
) -> tuple[float, float]:
    """Get frequency range for plotting PSD.
    
    Parameters
    ----------
    freq_range : ArrayLike | float
        If float, use (0, freq_range) Hz.
        If ArrayLike, use (freq_range[0], freq_range[1]) Hz.
        if ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    frequencies : ArrayLike | None
        Frequency array for getting frequency range.
    log_frequency : bool
        Whether using log frequency scale (remove 0 Hz from range if True).

    Returns
    -------
    tuple[float, float]
        Frequency range for plotting.
    """
    if frequencies is None:
        freq_limits = (0.1 if log_frequency else 0., np.inf)
    else:
        frequencies = np.asarray(frequencies)
        if log_frequency and frequencies[0] == 0:
            freq_limits = (frequencies[1], frequencies[-1])
        else:
            freq_limits = (frequencies[0], frequencies[-1])
    freq_range = np.asarray(freq_range, dtype=float)
    if freq_range.size == 0:
        freq_range = (0., np.inf)  # full frequency range
    elif freq_range.size == 1:
        freq_range = (0, freq_range.item())
    freq_range = (max(freq_range[0], freq_limits[0]), min(freq_range[1], freq_limits[1]))
    return freq_range


def fit_fooof(
    psd_da : xr.DataArray | ArrayLike,
    f : ArrayLike | None = None,
    freq_range : ArrayLike | float = (),
    aperiodic_mode : str = 'knee',
    peak_width_limits : ArrayLike = (0., np.inf),
    max_n_peaks : int = 10,
    dB_threshold : float = 0.8,
    peak_threshold : float = 1.0,
    report : bool = False
):
    """Fit PSD using FOOOF

    See https://github.com/fooof-tools/fooof for more details about the parameters.

    Parameters
    ----------
    psd_da : xr.DataArray | ArrayLike
        Power spectral density array. Dimension should be 'frequency'.
    f : ArrayLike | None
        Frequency array. Used only when psd_da is not a DataArray. Otherwise ignored.
    freq_range : ArrayLike | float
        Frequency range to fit. If float, use (0, freq_range) Hz.
        If ArrayLike, use (freq_range[0], freq_range[1]) Hz.
        If ArrayLike is empty, use (0, np.inf) Hz (full frequency range).
    aperiodic_mode : str
        Aperiodic mode. 'knee' or 'fixed'.
    peak_width_limits : ArrayLike
        Peak width limits. Minimum is 2 x frequency resolution.
        Maximum is the limit of the frequency range.
    max_n_peaks : int
        Maximum number of peaks.
    dB_threshold : float
        dB threshold.
    peak_threshold : float
        Peak threshold, in terms of the standard deviation of the aperiodic-removed power spectrum.
    report : bool
        Whether to print the report.

    Returns
    -------
    results : FOOOFResults
        FOOOF results.
    fooof : FOOOF
        FOOOF object.
    fooof_kwargs : dict
        Keyword arguments used to initialize the FOOOF object.
    """
    if isinstance(psd_da, xr.DataArray):
        pxx = psd_da.values
        f = psd_da.coords['frequency'].values
    else:
        pxx = np.asarray(psd_da)
        f = np.asarray(f)

    if aperiodic_mode != 'knee':
        aperiodic_mode = 'fixed'

    freq_range = _get_psd_freq_range(freq_range, frequencies=f, log_frequency=True)
    peak_width_limits = list(peak_width_limits)
    # minimum peak width is 2 x frequency resolution
    peak_width_limits[0] = max(peak_width_limits[0], (f[1] - f[0]) * 2)
    # maximum peak width is the limit of the frequency range
    peak_width_limits[1] = min(peak_width_limits[1], freq_range[1] - freq_range[0])

    # Initialize a FOOOF object
    fooof_kwargs = dict(
        peak_width_limits=peak_width_limits,
        min_peak_height=dB_threshold / 10,  # convert dB to log scale (base 10) power
        peak_threshold=peak_threshold,
        max_n_peaks=max_n_peaks,
        aperiodic_mode=aperiodic_mode,
    )
    fooof = FOOOF(**fooof_kwargs)  # Fooof model

    # Fit the model
    try:
        fooof.fit(f, pxx, freq_range)
    except Exception:  # exception due to uneven frequency spacing
        fl = np.linspace(f[0], f[-1], int((f[-1] - f[0]) / np.min(np.diff(f))) + 1)
        f, pxx = fl, np.interp(fl, f, pxx)
        fooof.fit(f, pxx, freq_range)
    results = fooof.get_results()

    if aperiodic_mode=='knee':
        ap_params = results.aperiodic_params
        if ap_params[1] <= 0:
            knee_freq = np.abs(ap_params[1]) ** (1 / ap_params[2])
            print(f"Knee location: {knee_freq:.2f} Hz")
            warnings.warn("Negative value of knee parameter occurred. Re-fit without knee parameter.")

            # re-fit without knee parameter
            fooof_kwargs['aperiodic_mode'] = 'fixed'
            fooof = FOOOF(**fooof_kwargs)
            fooof.fit(f, pxx, freq_range)
            results = fooof.get_results()

    if report:
        fooof.print_results()

    return results, fooof, fooof_kwargs


def get_fooof_freq_band(
    fooof_result : FOOOFResults,
    freq_range : tuple[float, float],
    top_n_peaks : int = 1,
    bandwidth_n_sigma : float = 1.5
) -> tuple[float, float]:
    """Get frequency band of the top N peaks in the FOOOF results within a given band of interest.
    
    Parameters
    ----------
    fooof_result : FOOOFResults
        FOOOF results.
    freq_range : tuple[float, float]
        Frequency band of interest
    top_n_peaks : int
        Number of top peaks to include in the band.
    bandwidth_n_sigma : float
        Multiplier of sigma of the Gaussian parameters to define the bandwidth of the peak.

    Returns
    -------
    band : tuple[float, float]
        Combined frequency band of the top N peaks within the given band of interest.
        If no peaks are found within the given band of interest, return None.
    peak_inds : array_like of bool
        Boolean array of the peaks within the given band of interest.
    """
    gaussian_params = fooof_result.gaussian_params
    peak_inds = (gaussian_params[:, 0] >= freq_range[0]) & (gaussian_params[:, 0] <= freq_range[1])
    band_peaks = gaussian_params[peak_inds, :]
    if band_peaks.size == 0:
        return None, peak_inds

    top_n_peaks = max(top_n_peaks, 1)  # at least one peak
    band_peaks = band_peaks[np.argsort(band_peaks[:, 1])[::-1][:top_n_peaks]]
    band_widths = bandwidth_n_sigma * band_peaks[:, 2]  # one-sided bandwidth
    band_freqs = np.fmax(band_peaks[:, [0]] + np.outer(band_widths, [-1, 1]), 0.)
    band = (band_freqs[:, 0].min(), band_freqs[:, 1].max())  # combined frequency band
    return band, peak_inds

