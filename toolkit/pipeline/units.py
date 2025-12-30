import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ttest_rel


OPTO_EVOKE_KEYS = ['opto_baseline_rate', 'opto_evoked_rate', 'opto_t_stat', 'opto_p_value']

def evoke_rate_test(
    spike_rate : xr.DataArray,
    baseline_window : tuple[float, float],
    evoked_window : tuple[float, float],
) -> pd.DataFrame:
    """Calculate evoked rate and perform significance t-test for units.
    
    Parameters
    ----------
    spike_rate : xr.DataArray
        Spike rate data array with coordinates: 'unit_id', 'time_relative_to_stimulus_onset', 'trial_id'.
    baseline_window : tuple[float, float]
        Baseline time window relative to stimulus onset.
    evoked_window : tuple[float, float]
        Evoked time window relative to stimulus onset.

    Returns
    -------
    evoke_df : pd.DataFrame
        Dataframe containing the evoked rate and t-test results. Index: 'unit_id'.
        Columns: 'opto_baseline_rate', 'opto_evoked_rate', 'opto_t_stat', 'opto_p_value'.
    """
    unit_ids = spike_rate.coords['unit_id'].values
    average_dims = ('time_relative_to_stimulus_onset', 'trial_id')
    # Count average over baseline and evoked time windows
    baseline_rates = spike_rate.sel(time_relative_to_stimulus_onset=slice(*baseline_window)).mean(dim=average_dims[0])
    evoked_rates = spike_rate.sel(time_relative_to_stimulus_onset=slice(*evoked_window)).mean(dim=average_dims[0])
    # Rate average over time windows and trials
    opto_baseline_rate = baseline_rates.mean(dim=average_dims[1])
    opto_evoked_rate = evoked_rates.mean(dim=average_dims[1])

    # t-test for each unit whether the evoked response is significantly greater than the baseline
    opto_t_stat = []
    opto_p_value = []
    for unit_id in unit_ids:
        t_stat, p_value = ttest_rel(evoked_rates.sel(unit_id=unit_id), baseline_rates.sel(unit_id=unit_id), alternative='greater')
        opto_t_stat.append(t_stat)
        opto_p_value.append(p_value)

    evoke_df = pd.DataFrame(dict(
        opto_baseline_rate=opto_baseline_rate,
        opto_evoked_rate=opto_evoked_rate,
        opto_t_stat=np.array(opto_t_stat),
        opto_p_value=np.array(opto_p_value)
    ), index=unit_ids)
    return evoke_df


def detect_optotag(
    units_info_df : pd.DataFrame,
    min_rate : float = 1.,
    evoked_ratio_threshold : float | None = 1.5,
    ttest_alpha : float | None = 0.05,
    spike_width_range : tuple[float, float] | None = None):
    """Detect optotag units using evoked ratio and spike width range from dataframe.
    
    Parameters
    ----------
    units_info_df : pd.DataFrame
        Dataframe containing the units information.
    min_rate : float, optional
        Minimum rate to avoid 0 firing rate for evoked ratio calculation.
    evoked_ratio_threshold : float, optional
        Threshold for evoked ratio. If None, no threshold is applied.
    ttest_alpha : float, optional
        Alpha for t-test. If None, no t-test is performed.
    spike_width_range : tuple[float, float], optional
        Range for spike width. If None, no spike width range limit is applied.

    Returns
    -------
    optotag_df : pd.DataFrame
        Copy of the input dataframe with the optotagging results added.
    positive_units : np.ndarray
        Array of unit IDs that are positive for optotagging.
    """
    optotag_df = units_info_df.copy()
    optotag_df['evoked_ratio'] = (optotag_df['opto_evoked_rate'] + min_rate) / (optotag_df['opto_baseline_rate'] + min_rate)
    if evoked_ratio_threshold is None:
        optotag_df['evoke_positive'] = True
    else:
        optotag_df['evoke_positive'] = optotag_df['evoked_ratio'] > evoked_ratio_threshold
    if ttest_alpha is None:
        optotag_df['evoke_significant'] = True
    else:
        optotag_df['evoke_significant'] = ~optotag_df['opto_p_value'].isna() & (optotag_df['opto_p_value'] < ttest_alpha)
    if spike_width_range is None:
        optotag_df['valid_spike_width'] = True
    else:
        optotag_df['valid_spike_width'] = (optotag_df['waveform_duration'] >= spike_width_range[0]) \
            & (optotag_df['waveform_duration'] <= spike_width_range[1])
    optotag_df['optotag_positive'] = optotag_df['evoke_positive'] & optotag_df['evoke_significant'] & optotag_df['valid_spike_width']
    positive_units = optotag_df.index[optotag_df['optotag_positive']].values
    return optotag_df, positive_units
