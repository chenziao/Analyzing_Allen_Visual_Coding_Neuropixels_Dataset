"""
Batch process sessions for

- Calculate average PSD of stimuli.
- Calculate PSD averaged across each condition of drifting gratings stimuli.
"""

import add_path

PARAMETERS = dict(
    group_width = dict(
        default = 1,
        type = int,
        help = "Number of channels to the left and right of the central channel to average LFP over."
    ),
    psd_tseg = dict(
        default = 0.5,
        type = float,
        help = "Segment duration in seconds for PSD Welch method."
    ),
    df = dict(
        default = 1.0,
        type = float,
        help = "Frequency resolution in Hz for PSD."
    )
)


def process_stimuli_psd(
    session_id: int,
    group_width: int = 1,
    psd_tseg: float = 0.5,
    df: float = 1.0,
):
    import toolkit.allen_helpers.stimuli as st
    import toolkit.pipeline.signal as ps
    from toolkit.pipeline.data_io import SessionDirectory
    from toolkit.pipeline.global_settings import GLOBAL_SETTINGS

    #################### Get session and probe ####################
    ecephys_structure_acronym = GLOBAL_SETTINGS.get('ecephys_structure_acronym')

    session_dir = SessionDirectory(session_id, ecephys_structure_acronym, cache_lfp=True)

    probe_info = session_dir.load_probe_info()
    if not session_dir.has_lfp_data:  # Skip session if it has no LFP data
        print(f"Session {session_id} has no LFP data. Skipping...")
        return

    session = session_dir.session

    lfp_groups, channel_groups = ps.get_lfp_channel_groups(
        session_dir, probe_info['central_channels'], width=group_width
    )


    #################### Analyze data ####################
    stimulus_presentations = session.stimulus_presentations
    session_type = session.session_type

    stimulus_names = st.STIMULUS_NAMES[session_type]
    drifting_gratings_stimuli = st.STIMULUS_CATEGORIES[session_type]['drifting_gratings']

    # Process PSD
    psd_das = {}
    cond_psd_das = {}

    for stim in stimulus_names:
        stim_trials = st.get_stimulus_trials(stimulus_presentations, stimulus_name=stim)
        aligned_lfp = st.align_trials(lfp_groups, stim_trials, window=(0., stim_trials.duration))
        psd_trials = ps.trial_psd(aligned_lfp, tseg=psd_tseg, df=df)
        psd_avg = psd_trials.mean(dim='presentation_id', keep_attrs=True)
        psd_das[stim] = psd_avg

        if stim in drifting_gratings_stimuli:  # has conditions
            conditions = st.presentation_conditions(stim_trials.presentations, condition_types=st.CONDITION_TYPES)
            cond_psd = st.average_across_conditions(psd_trials, *conditions)
            cond_psd_das[stim] = cond_psd


    #################### Save data ####################
    # Save PSD of stimuli
    session_dir.save_psd(psd_das, channel_groups)

    # Save conditions PSD of drifting gratings
    session_dir.save_conditions_psd(cond_psd_das)


if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    process_sessions(process_stimuli_psd, **args)

