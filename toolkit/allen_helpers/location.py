from pathlib import Path
import numpy as np

CCF_COORDS = [
    'anterior_posterior_ccf_coordinate',
    'dorsal_ventral_ccf_coordinate',
    'left_right_ccf_coordinate'
]


def get_lfp_channel_positions(cache_dir, session_id, probe_id):
    """
    Return electrode positions for the LFP channels of a given probe,
    aligned to the channel order in session.get_lfp(probe_id).

    Parameters
    ----------
    cache_dir : str | Path
        Base cache directory where NWB files are stored.
    session_id : int
        Ecephys session ID.
    probe_id : int
        Probe ID.

    Returns
    -------
    DataFrame
        Indexed by 'channel' (matches lfp xarray 'channel').
        Columns include ['local_index','probe_vertical_position',
                         'probe_horizontal_position','x','y','z',
                         'group_name','location'].

    Notes
    -----
    The 'x', 'y', 'z' coordinates in NWB file correspond to the CCF coordinates
    'anterior_posterior', 'dorsal_ventral', 'left_right', respectively.
    However, the 'z' coordinate is mistakenly replicated from the 'y' coordinate.
    Use with caution.
    """
    lfp_path = Path(cache_dir) / f"session_{session_id}" / f"probe_{probe_id}_lfp.nwb"
    if not lfp_path.exists():
        raise FileNotFoundError(f"LFP file not found at {lfp_path}")

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(lfp_path, 'r') as io:
        nwb = io.read()
        # Grab the LFP ElectricalSeries
        es = nwb.acquisition[f"probe_{probe_id}_lfp_data"]
        # Indices of the LFP channels into the file's electrodes table
        lfp_elec_idx = np.asarray(es.electrodes.data)
        # Subset electrodes table in the given order
        elec_df = nwb.electrodes.to_dataframe().iloc[lfp_elec_idx].copy()

    # Rename 'id' to 'channel' and set as index
    elec_df = elec_df.reset_index().rename(columns={'id': 'channel'}).set_index('channel')
    return elec_df


