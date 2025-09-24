# Analyzing_Allen_Visual_Coding_Neuropixels_Dataset

## Resource
This repository is for analyzing the public dataset https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html.

## Requirements

python >= 3.10  (legacy: 3.8.16)

## Analysis Procedures (Legacy)

1. Edit the [configuration file](config.json) to set directories of `cache_dir` for allensdk data cache, `output_dir` for result data, and `figure_dir` for result figures. Specify properties in `filter_dict` for filtering sessions in the dataset.

2. Run [Choose_Visual_Data](Choose_Visual_Data.ipynb) to retrieve data of a target region in a session. Local field potential (LFP) data averaged from groups of channels and an information json file of the selected session and region will be saved to the output folder.

3. Edit `analysis_object` in the [configuration file](config.json) to refer to the selected session and region. Set `enter_parameters` to `true` for the first time analyzing the selected object.

4. Run [Analyze_Visual_LFP](Analyze_Visual_LFP.ipynb) to analyze power spectrum of the LFP.

5. Run [Analyze_Visual_Spike](Analyze_Visual_Spike.ipynb) to analyze population firing activities and save result data to the output folder.

6. Run [Analyze_Visual_Entrainment](Analyze_Visual_Entrainment.ipynb) to analyze entrainment of units to the oscillation and correlation between properties of units. Step 4 and 5 need to be done before this step.

7. The selected parameters for the analyses are saved in the information json file. If `enter_parameters` is set to `false` and analyses from step 4 to 6 are run again, the saved parameters will be used. If `save_figure` is set to `true`, figures will be saved to the figure folder.
