"""
Population vector during stimuli (during drifting gratings and natural movies).

- Compute normalized and downsampled LFP band power during stimuli.
- Get spike rate of units during stimuli.
- Get firing rate statistics of units and compute normalized firing rate.
- Perform PCA and compute population vector.
- Save data.
"""

import add_path


PARAMETERS = dict(
    overwrite_spike_counts = dict(
        default = False,
        type = bool,
        help = "Whether to overwrite `units_spike_rate` data file."
    ),
    combine_stimulus_name = dict(
        default = 'mixed_stimuli',
        type = str,
        help = "Stimuli combination for population vector analysis. "
            "'natural_movies' or 'drifting_gratings' or 'mixed_stimuli'."
    ),
    filter_orientation = dict(
        default = False,
        type = bool,
        help = "Whether to filter conditions by preferred orientation."
    ),
    sigma = dict(
        default = 0,
        type = float,
        help = "Gaussian filter sigma in seconds for smoothing firing rate." 
            "If 0, set to bin width."
    ),
    normalize_unit_fr = dict(
        default = True,
        type = bool,
        help = "Whether to normalize unit firing rate."
    ),
    soft_normalize = dict(
        default = True,
        type = bool,
        help = "Whether to use soft normalization."
    ),
    normalization_scale = dict(
        default = 'std',
        type = str,
        help = "Normalization scale for unit firing rate. 'max', 'std', 'mean'."
    ),
    quantile = dict(
        default = 0.2,
        type = float,
        help = "Quantile for soft normalization cutoff."
    ),
    select_RS = dict(
        default = True,
        type = bool,
        help = "Whether to select only Regular Spiking (RS) units."
    ),
    select_layer = dict(
        default = False,
        type = bool,
        help = "Whether to select units the layer of interest."
    ),
    absolute_origin = dict(
        default = True,
        type = bool,
        help = "Whether to use absolute zeros firing rate as the origin in principal space."
    ),
    n_pc_range = dict(
        default = [3, 6],
        type = list,
        help = "Range of number of principal components to consider."
    )
)

def population_vector_during_stimuli(
    session_id: int,
    overwrite_spike_counts: bool = False,
    combine_stimulus_name: str = 'mixed_stimuli',
    filter_orientation: bool = False,
    sigma: float = 0,
    normalize_unit_fr: bool = True,
    soft_normalize: bool = True,
    normalization_scale: str = 'std',
    quantile: float = 0.2,
    select_RS: bool = True,
    select_layer: bool = False,
    absolute_origin: bool = True,
    n_pc_range: list[int] = [3, 6],
    is_main_job: bool = True,
) -> None:

    import pandas as pd
    import xarray as xr

    pass



if __name__ == "__main__":
    from toolkit.pipeline.batch_process import BatchProcessArgumentParser, process_sessions

    parser = BatchProcessArgumentParser(parameters=PARAMETERS)
    args = parser.parse_args()

    # Determine if running as main job
    array_index = args['array_index']
    args['parameters']['is_main_job'] = array_index is None or array_index == 0

    process_sessions(population_vector_during_stimuli, **args)
