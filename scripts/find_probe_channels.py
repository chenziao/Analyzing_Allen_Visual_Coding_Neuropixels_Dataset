"""
Batch process sessions for

- Initial processing: download and cache data from Allen Database.
- Find probe channels for the target structure (e.g. VISp).
- Compute CSD for the channels in the structure.
"""

import add_path

PARAMETERS = dict(
    csd_sigma_time = dict(
        default = 1.6,
        type = float,
        help = "Temporal Gaussian sigma in ms."
    ),
    csd_sigma_space = dict(
        default = 40.0,
        type = float,
        help = "Spatial Gaussian sigma in µm."
    ),
    cache_data_only = dict(
        default = False,
        type = bool,
        help = "Only cache data, skip further processing."
    ),
    timeout = dict(
        default =1800,
        type = int,
        help = "Timeout for halting downloads in seconds. "
            "Set to 0 to disable timeout."
    )
)


def find_probe_channels(
    session_id: int,
    csd_sigma_time: float = 1.6,
    csd_sigma_space: float = 40.0,
    cache_data_only: bool = False,
    timeout: int = 1800
):
    import numpy as np
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

    from toolkit import paths
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS
    from toolkit.allen_helpers.location import CCF_COORDS
    from toolkit.allen_helpers.location import fill_missing_linear_channels, StructureFinder, central_channel_in_structure
    from toolkit.analysis.signal import compute_csd
    from toolkit.pipeline.data_io import SessionDirectory

    from toolkit.utils.timeout_handler import run_with_timeout

    #################### Get session and probe ####################
    cache = EcephysProjectCache.from_warehouse(manifest=paths.ECEPHYS_MANIFEST_FILE)
    ecephys_structure_acronym = GLOBAL_SETTINGS.get('ecephys_structure_acronym')

    # get channels in the structure
    def download_session():
        return cache.get_session_data(session_id)

    session = run_with_timeout(download_session, timeout, f"Session {session_id} data download")

    all_channels = session.channels
    channels = all_channels.loc[all_channels['structure_acronym'] == ecephys_structure_acronym]

    # get probes for the session and structure (usually only one)
    probes = cache.get_probes()
    probes = probes[probes['ecephys_session_id'] == session_id]
    probes = probes.iloc[
        [ecephys_structure_acronym in x for x in probes['ecephys_structure_acronyms']]
    ]

    # get the probe with the most channels in the structure
    n_channels_in_probe =  [np.count_nonzero(channels['probe_id'] == i) for i in probes.index]
    probe_id = probes.index[np.argmax(n_channels_in_probe)]
    fs = probes.loc[probe_id, 'lfp_sampling_rate']

    # channels in the probe sorted by probe_channel_number
    probe_channels = all_channels.loc[all_channels['probe_id'] == probe_id].sort_values('probe_channel_number')

    # overview of channels layout in the structure
    vertical_position_range = channels['probe_vertical_position'].max() - channels['probe_vertical_position'].min()
    n_missing_channels = channels['probe_channel_number'].max() - channels['probe_channel_number'].min() + 1 - len(channels)
    print(f"Number of channels: {len(channels):d}")
    print(f"Number of missing channels in middle: {n_missing_channels:d}")
    print(f"Vertical range: {vertical_position_range:d} μm")
    print(f"Number of rows: {vertical_position_range // 20 + 1:d}")

    # Load LFP given probe
    def download_lfp():
        return session.get_lfp(probe_id)

    lfp_array = run_with_timeout(download_lfp, timeout, f"Probe {probe_id} LFP download")
    lfp_array.attrs.update(fs=fs)

    if cache_data_only:
        print("Cache data only, skipping further processing.")
        return


    #################### Find LFP channels locations in structure ####################

    # fill possible missing channels
    probe_channels = fill_missing_linear_channels(probe_channels, lfp_array.channel)

    # get channels in the structure in the probe
    channels = probe_channels.loc[probe_channels['structure_acronym'] == ecephys_structure_acronym]

    # Ensure channels are sorted by vertical position
    channel_idx = np.argsort(probe_channels.loc[lfp_array.channel, 'probe_vertical_position'])
    if not np.all(np.diff(channel_idx) == 1):
        lfp_array = lfp_array.isel(channel=channel_idx)

    # get LFP channels in the structure
    channel_idx = np.nonzero([i in channels.index for i in lfp_array.channel.values])[0]
    channel_idx_csd = np.copy(channel_idx)  # channel indices for CSD

    padding = [1, 1]  # replicate padding for iCSD boundaries
    if channel_idx[0] > 0:
        padding[0] = 0  # no padding if a leading channel exists
        channel_idx_csd = np.insert(channel_idx_csd, 0, channel_idx[0] - 1)
    if channel_idx[-1] < lfp_array.channel.size - 1:
        padding[1] = 0  # no padding if a trailing channel exists
        channel_idx_csd = np.append(channel_idx_csd, channel_idx[-1] + 1)

    channels_id = lfp_array.channel[channel_idx]
    channels_id_csd = lfp_array.channel[channel_idx_csd]

    csd_channel_positions = probe_channels.loc[channels_id_csd, 'probe_vertical_position']

    # validate if the spacing between LFP channels is consistent
    if np.unique(np.diff(csd_channel_positions)).size != 1:
        raise ValueError('The spacing between LFP channels is not consistent')

    # Get dataframe for LFP channels in the structure and get layer of each channel
    lfp_channels = channels.loc[channels_id]
    sf = StructureFinder(paths.REFERENCE_SPACE_CACHE_DIR)
    lfp_channels['layer_acronym'], lfp_channels['inside_structure'] = \
        sf.get_structure_array(lfp_channels[CCF_COORDS])

    # Get central channels in each layer
    valid_channels = lfp_channels.loc[lfp_channels['inside_structure']]
    central_channels = central_channel_in_structure(valid_channels['layer_acronym'])
    central_channels = {s: int(valid_channels.index[i]) for s, i in central_channels.items()}

    # Select channels for CSD
    csd_array = compute_csd(
        lfp=lfp_array.sel(channel=channels_id_csd),
        positions=csd_channel_positions,
        sigma_time=csd_sigma_time,
        sigma_space=csd_sigma_space,
        padding=padding
    )


    #################### Save data ####################
    # Get session cache directory
    session_dir = SessionDirectory(session_id, ecephys_structure_acronym)

    # Save LFP channels info
    session_dir.save_lfp_channels(lfp_channels)

    # Save probe info
    session_dir.save_probe_info(probe_id, central_channels, channels_id_csd, padding)

    # Save CSD
    session_dir.save_csd(csd_array)


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(find_probe_channels, **args)

