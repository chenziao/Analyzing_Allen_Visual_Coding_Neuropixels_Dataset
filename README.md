# Analyzing_Allen_Visual_Coding_Neuropixels_Dataset

## Resource
This repository is for analyzing the public dataset https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html.

## Requirements

python >= 3.10  (legacy: 3.8.16)

Requires Anaconda (conda). Create and use a dedicated environment:

```bash
# create env with Python >=3.10 and the Anaconda metapackage
conda create -n allen python=3.10 anaconda -y

# activate the environment
conda activate allen

# install project dependencies into the conda env
pip install -r requirements.txt
```

## Modules

- [Notebooks](notebooks): Jupyter notebooks for analyzing and visualizing.

- [Scripts](scripts): Scripts for batch processing with automated pipeline.

- [Tools](tools): Test notebooks for developing tool functions.

- [Toolkit](toolkit): Function modules for this repo.

- [Docs](docs): Documentation for this repo.

- [Legacy](legacy): Legacy code for this repo.

## Configuration

- [path_config.json](path_config.json): Set the paths for the cache data and output data.

- [global_settings.json](global_settings.json): Set the global settings and parameters for the analysis.

- [output_config.json](output_config.json): Set the format for the output data.

- [sessions.json](sessions.json): List of session IDs for test run of batch processing scripts and sessions blacklist to exclude sessions to process.

## Analysis Procedures

### Setup on the server

- See [Requirements](#requirements) for how to create a conda environment with the necessary dependencies.

- Set the paths for the cache data and output data in [path_config.json](path_config.json). Suggest using the shared directory on the server `/home/shared/Allen_Visual_Coding_Data` as the root directory.

- Make sure the conda environment is activated before running any script.

- Edit the batch script [batch_run_script.sh](batch_run_script.sh) to run the desired python script under folder `scripts/`. Set the argument `--session_set` to the desired set of sessions to process. Set other arguments if needed. Run `python scripts/[script_name].py -h` to see the available arguments for the script.

- Make sure directory `./stdout/` exists under the root directory of the repository. This is the directory for the output logs of the batch processing scripts.

- Run the batch script `sbatch batch_run_script.sh` to process the sessions.

### Scripts (for batch processing)

- After a script finishes running, check the `batch_logs` folder (see `batch_log_dir` in [path_config.json](path_config.json)) for the logs of printed messages and errors. The parameters used for the script are saved in `.json` files in the `batch_logs` folder.

- Common arguments for all scripts:
    - `--session_set`: The session set to process the sessions from. Available sets: `all`, `test`, `selected`, `optotag`.
      - `all`: All sessions in Allen's database.
      - `test`: Test sessions listed in `'test'` key of [sessions.json](sessions.json).
      - `selected`: Selected sessions recorded in `session_selection.csv` file in the `output` folder (see `output_base_dir` in [path_config.json](path_config.json)).
      - `optotag`: Optotag sessions. (Not implemented yet)
    - `--session_list`: List of session IDs to process (space-separated). `--session_set` argument will be ignored if this is provided.
    - `--use_blacklist`: Use sessions blacklist to exclude sessions to process to avoid uncaught errors in some sessions that may cause the batch processing to stall. The blacklist is listed in `'blacklist'` key of [sessions.json](sessions.json).
    - `--disable_logging`: Disable logging to the log file.

#### Scripts (execute in order, later scripts may depend on the results of previous scripts)

1. [find_probe_channels.py](scripts/find_probe_channels.py)

    - Initial processing: download and cache data from Allen Database. If `--cache_data_only` is set to `True`, the script will only perform this step and skip further processing.
    - Find probe channels for the target structure (e.g. VISp).
    - Compute CSD for the channels in the structure. If `--skip_compute_csd` is set to `True`, the script will skip computing CSD.

    Note: Run the notebook [check_channel_layer_positions](notebooks/check_channel_layer_positions.ipynb) to overwrite the probe info file and LFP channels file whose layer positions are missing to avoid errors in further processing.

2. [process_stimuli_psd.py](scripts/process_stimuli_psd.py)

    - Calculate average PSD of stimuli.
    - Calculate PSD averaged across each condition of drifting gratings stimuli.

### Notebooks (for interactive analysis and visualization)

- [Find_Probe_Channels](notebooks/Find_Probe_Channels.ipynb)

  Initial processing before analyzing a session. Find the LFP channels in the selected cortical structure and get the central channels in each layer.

- [check_channel_layer_positions](notebooks/check_channel_layer_positions.ipynb)

  Check the layer positions of the LFP channels in the selected structure. Some sessions may have LFP channels with missing CCF coordinates. The layer positions are guessed by the vertical position of the channels according to the average portion of boundaries between layers estimated from the sessions with CCF coordinates. Overwrite the probe info file and LFP channels file with guessed layer for the channels.

- [Spectral_analysis](notebooks/Spectral_analysis.ipynb)

  Calculate PSD of stimuli of a session and apply FOOOF to fit the PSD.

- [CSD_during_stimuli](notebooks/CSD_during_stimuli.ipynb)

  Analyze the CSD during stimuli.

## Analysis Procedures (Legacy)

1. Edit the [configuration file](config.json) to set directories of `cache_dir` for allensdk data cache, `output_dir` for result data, and `figure_dir` for result figures. Specify properties in `filter_dict` for filtering sessions in the dataset.

2. Run [Choose_Visual_Data](Choose_Visual_Data.ipynb) to retrieve data of a target region in a session. Local field potential (LFP) data averaged from groups of channels and an information json file of the selected session and region will be saved to the output folder.

3. Edit `analysis_object` in the [configuration file](config.json) to refer to the selected session and region. Set `enter_parameters` to `true` for the first time analyzing the selected object.

4. Run [Analyze_Visual_LFP](Analyze_Visual_LFP.ipynb) to analyze power spectrum of the LFP.

5. Run [Analyze_Visual_Spike](Analyze_Visual_Spike.ipynb) to analyze population firing activities and save result data to the output folder.

6. Run [Analyze_Visual_Entrainment](Analyze_Visual_Entrainment.ipynb) to analyze entrainment of units to the oscillation and correlation between properties of units. Step 4 and 5 need to be done before this step.

7. The selected parameters for the analyses are saved in the information json file. If `enter_parameters` is set to `false` and analyses from step 4 to 6 are run again, the saved parameters will be used. If `save_figure` is set to `true`, figures will be saved to the figure folder.
