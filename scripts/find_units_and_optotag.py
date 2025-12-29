"""
Batch process sessions for

- Find units in the target structure.
- Test the evoked rate of units using t-test with optotagging trials.
- Save selected units info with optotagging results.
"""

import add_path

import numpy as np
import pandas as pd
import xarray as xr
from toolkit.analysis.utils import get_bins
from toolkit.analysis.spikes import spike_count


PARAMETERS = dict(
    nearby_channel_min_distance = dict(
        default = 200.,
        type = float,
        help = "Nearby channel at least this distance apart in µm."
    ),
    min_vertical_distance = dict(
        default = 10.,
        type = float,
        help = "Minimum vertical distance to consider as the same vertical position in µm."
    ),
    bin_width = dict(
        default = 0.0005,
        type = float,
        help = "Bin width in seconds for counting spikes."
    ),
    baseline_window = dict(
        default = (-0.010, -0.001),
        type = tuple,
        help = "Baseline time window relative to opto stimulus onset in seconds."
    ),
    evoked_window = dict(
        default = (0.001, 0.010),
        type = tuple,
        help = "Evoked time window relative to opto stimulus onset in seconds."
    )
)


def optotagging_spike_rate(spike_times, opto_trials, unit_ids, bin_width=0.03, window=(-1.0, 1.0)):
    """Count spikes in bins for each optotagging trial"""
    bin_centers, bin_edges = get_bins(window, bin_width=bin_width)
    trial_start_times = np.asarray(opto_trials['start_time'])
    unit_ids = np.asarray(unit_ids)

    units_spk_counts = np.zeros((trial_start_times.size, bin_centers.size, unit_ids.size), dtype=int)
    for i, unit_id, in enumerate(unit_ids):
        tspk = spike_times[unit_id]
        for j, trial_start in enumerate(trial_start_times):
            t = tspk[(tspk >= trial_start + bin_edges[0]) & (tspk <= trial_start + bin_edges[-1])]
            units_spk_counts[j, :, i] = spike_count(t - trial_start, bin_edges)

    units_spk_rate = xr.DataArray(data=units_spk_counts / bin_width, coords=dict(
        trial_id=opto_trials.index.values, time_relative_to_stimulus_onset=bin_centers, unit_id=unit_ids),
        name='spike_rate', attrs=dict(bin_width=bin_width))
    return units_spk_rate


def find_units_and_optotag(
    session_id: int,
    nearby_channel_min_distance: float = 200.,
    min_vertical_distance: float = 10.,
    bin_width: float = 0.0005,
    baseline_window: tuple[float, float] = (-0.010, -0.001),
    evoked_window: tuple[float, float] = (0.001, 0.010),
) -> None:
    from functools import lru_cache
    from toolkit.pipeline.data_io import SessionDirectory
    from toolkit.utils.misc import pd_merge_differences
    from toolkit.pipeline.units import evoke_rate_test

    #################### Get session and probe ####################
    session_dir = SessionDirectory(session_id)

    probe_info = session_dir.load_probe_info()
    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    lfp_channels = session_dir.load_lfp_channels()

    session = session_dir.session

    #################### Analyze data ####################
    # Get units dataframe
    units = session.units
    sel_units_idx = units['probe_id'] == probe_info['probe_id']
    sel_units_idx = sel_units_idx & (units['ecephys_structure_acronym'] == session_dir.structure_acronym)
    sel_units = units[sel_units_idx]

    # Get nearby LFP channels
    offsets = dict(
        lfp_channel = 0.,
        lower_lfp_channel = -nearby_channel_min_distance,
        upper_lfp_channel = nearby_channel_min_distance
    )

    @lru_cache(maxsize=128)
    def nearest_lfp_channel(x):
        """Return the index of the nearest LFP channel to vertical position x and if a channel exists at x."""
        d = (lfp_channels['probe_vertical_position'] - x).abs()
        idx = d.idxmin()
        return idx, d.loc[idx] <= min_vertical_distance

    units_lfp_channels = []
    for x in sel_units['probe_vertical_position'].values:
        unit_lfp_channels = {}
        for s, offset in offsets.items():
            unit_lfp_channels[s + '_id'], unit_lfp_channels[s + '_exists'] = nearest_lfp_channel(x + offset)
        units_lfp_channels.append(unit_lfp_channels)
    units_lfp_channels = pd.DataFrame(units_lfp_channels, index=sel_units.index)
    units_lfp_channels['layer_acronym'] = lfp_channels.loc[units_lfp_channels['lfp_channel_id'], 'layer_acronym'].values

    sel_units = pd_merge_differences(sel_units, units_lfp_channels, left_index=True, right_index=True, how='left')

    # Get optotagging info
    opto_epochs = session.optogenetic_stimulation_epochs
    # get trials with duration 10 ms (a single square pulse)
    trials = opto_epochs[(opto_epochs.duration > 0.009) & (opto_epochs.duration < 0.02)]
    window = (min(baseline_window[0], evoked_window[0]), max(baseline_window[1], evoked_window[1]))
    spike_rate = optotagging_spike_rate(session.spike_times, trials, sel_units.index, bin_width=bin_width, window=window)

    # Calculate evoked rate and perform significance t-test
    evoke_df = evoke_rate_test(spike_rate, baseline_window, evoked_window)
    evoke_df['genotype'] = session_dir.genotype
    sel_units = pd_merge_differences(sel_units, evoke_df, left_index=True, right_index=True, how='left')

    #################### Save data ####################
    # Save selected units info with optotagging results
    session_dir.save_units_info(sel_units)


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(find_units_and_optotag, **args)
