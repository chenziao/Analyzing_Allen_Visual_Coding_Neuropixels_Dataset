#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Run python sript if enter_parameters is needed.
# Run ipython or notebook with enter_parameters disabled if cell outputs need to be saved.
run_from_ipython = 'get_ipython' in globals()
# Use ipython display when running notebook. Print directly to console when running sript.
display = display if run_from_ipython else print


# In[2]:


import os
import json
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from utils import *
from process import *
from analysis import *
from plots import *

if run_from_ipython:
    get_ipython().run_line_magic('matplotlib', 'inline')

with open('config.json') as f:
    config = json.load(f)


# In[3]:


# Cache directory path, it determines where downloaded data will be stored
manifest_path = os.path.join(config['cache_dir'], "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Whether to manually enter parameters for analysis
enter_parameters = config['enter_parameters']
whether_redo = redo_condition(enter_parameters)


# ## Get session and probe

# In[4]:


session_id = config['analysis_object']['session_id']
ecephys_structure_acronym = config['analysis_object']['ecephys_structure_acronym']
session = cache.get_session_data(session_id)


# In[5]:


output_dir = config['output_dir']
session_dir = os.path.join(output_dir, f'session_{session_id:d}')
info_file = os.path.join(output_dir, f'session_{session_id:d}_{ecephys_structure_acronym:s}.json')
with open(info_file) as f:
    info = json.load(f)
print(json.dumps(info, indent=4))

probe_id = info['probe_id']
probe_dir = os.path.join(session_dir, f'probe_{probe_id:d}')
parameters_dict = info['parameters']
fig_disp = figure_display_function(config)


# ## Load LFP channels

# In[6]:


filepath = os.path.join(probe_dir, f'{ecephys_structure_acronym:s}_lfp_channel_groups.nc')
with xr.open_dataset(filepath) as f:
    lfp_array = f.load()
lfp_array = lfp_array.assign_attrs(fs=info['fs']).rename(group_id='channel')
display(lfp_array)


# In[7]:


channel_group_map = pd.read_csv(filepath.replace('.nc', '.csv'), index_col='id')
group_dv_ccf = dict(zip(channel_group_map['group_id'], channel_group_map['dorsal_ventral_ccf_coordinate']))
display(channel_group_map)


# In[8]:


plt.figure(figsize=(3, 2.4))
plt.plot(group_dv_ccf.keys(), group_dv_ccf.values(), marker='o')
plt.xlabel('Channel Group ID')
plt.ylabel('Dorsal Ventral CCF Coordinate')
fig_disp('channel_group_coordinates')


# ## Analysis

# In[9]:


stimulus_presentations = session.stimulus_presentations
channel_id = info['center_unit_channel_id']
group_id = channel_group_map.loc[channel_id, 'group_id']


# ### drifting_gratings

# In[10]:


redo = True
while redo:
    ap_knee, = get_parameters({'drifting_gratings_knee': True}, parameters_dict, enter_parameters)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    drifting_gratings_presentations, grating_ids, grating_times, grating_duration = align_gratings(
        stimulus_presentations, stimulus_name='drifting_gratings')
    aligned_lfp = align_trials(lfp_array, grating_ids, grating_times, window=(0., grating_duration))
    drifting_gratings_psd = trial_psd(aligned_lfp, tseg=1.)
    psd_avg = drifting_gratings_psd.mean(dim='presentation_id')
    _, fig1, fig2 = plot_channel_psd(psd_avg, channel_id=group_id, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                                   aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
    fig_disp({'drifting_grating_psd': fig1, f'drifting_grating_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# ### static_gratings

# In[11]:


redo = True
while redo:
    ap_knee, = get_parameters({'static_gratings_knee': False}, parameters_dict, enter_parameters)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    _, presentations_ids, presentations_times, trial_duration = align_gratings(stimulus_presentations, stimulus_name='static_gratings')
    aligned_lfp = align_trials(lfp_array, presentations_ids, presentations_times, window=(0., trial_duration))
    psd_avg = trial_psd(aligned_lfp, tseg=1.).mean(dim='presentation_id')
    _, fig1, fig2 = plot_channel_psd(psd_avg, channel_id=group_id, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                         aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
    fig_disp({'static_grating_psd': fig1, f'static_grating_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# ### natural_scenes

# In[12]:


redo = True
while redo:
    ap_knee, = get_parameters({'natural_scenes_knee': False}, parameters_dict, enter_parameters)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    _, presentations_ids, presentations_times, trial_duration = align_scenes(stimulus_presentations, stimulus_name='natural_scenes')
    aligned_lfp = align_trials(lfp_array, presentations_ids, presentations_times, window=(0., trial_duration))
    psd_avg = trial_psd(aligned_lfp, tseg=1.).mean(dim='presentation_id')
    _, fig1, fig2 = plot_channel_psd(psd_avg, channel_id=group_id, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                         aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
    fig_disp({'natural_scenes_psd': fig1, f'natural_scenes_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# ### natural_movie_one

# In[13]:


redo = True
while redo:
    ap_knee, = get_parameters({'natural_movie1_knee': True}, parameters_dict, enter_parameters)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    _, presentations_ids, presentations_times, trial_duration = align_movie(stimulus_presentations, stimulus_name='natural_movie_one')
    aligned_lfp = align_trials(lfp_array, presentations_ids, presentations_times, window=(0., trial_duration))
    psd_avg = trial_psd(aligned_lfp, tseg=1.).mean(dim='presentation_id')
    _, fig1, fig2 = plot_channel_psd(psd_avg, channel_id=group_id, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                         aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
    fig_disp({'natural_movie1_psd': fig1, f'natural_movie1_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# In[14]:


redo = True
while redo:
    channel_ids, = get_parameters({'scaleogram_channels': list(map(int, lfp_array.channel))}, parameters_dict, enter_parameters)

    sxx_avg = trial_averaged_spectrogram(aligned_lfp, tseg=.5)
    _ = plot_channel_spectrogram(sxx_avg, channel_id=channel_ids, plt_range=(0, 100.), plt_log=True,
                                 clr_freq_range=(30., 80.), figsize=(6, 3.6),
                                 remove_aperiodic={'freq_range': 200., 'aperiodic_mode': aperiodic_mode})
    fig_disp('natural_movie1_scaleogram')

    redo = whether_redo()


# ### natural_movie_three

# In[15]:


redo = True
while redo:
    ap_knee, = get_parameters({'natural_movie1_knee': True}, parameters_dict, enter_parameters)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    _, presentations_ids, presentations_times, trial_duration = align_movie(stimulus_presentations, stimulus_name='natural_movie_three')
    aligned_lfp = align_trials(lfp_array, presentations_ids, presentations_times, window=(0., trial_duration))
    psd_avg = trial_psd(aligned_lfp, tseg=1.).mean(dim='presentation_id')
    _, fig1, fig2 = plot_channel_psd(psd_avg, channel_id=group_id, freq_range=200., plt_range=(0, 100.), figsize=(5, 4),
                         aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
    fig_disp({'natural_movie3_psd': fig1, f'natural_movie3_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# In[16]:


redo = True
while redo:
    channel_ids, = get_parameters({'scaleogram_channels': list(map(int, lfp_array.channel))}, parameters_dict, enter_parameters)

    sxx_avg = trial_averaged_spectrogram(aligned_lfp, tseg=.5)
    _ = plot_channel_spectrogram(sxx_avg, channel_id=channel_ids, plt_range=(0, 100.), plt_log=True,
                                 clr_freq_range=(30., 80.), figsize=(6, 3.6),
                                 remove_aperiodic={'freq_range': 200., 'aperiodic_mode': aperiodic_mode})
    fig_disp('natural_movie3_scaleogram')

    redo = whether_redo()


# ### drifting_gratings conditions

# In[17]:


condition_id, cond_presentation_id = presentation_conditions(
    drifting_gratings_presentations, condtion_types=['orientation', 'temporal_frequency'])

drifting_gratings_cond_psd = [drifting_gratings_psd.sel(presentation_id=i).mean(dim='presentation_id') for i in cond_presentation_id.values()]
drifting_gratings_cond_psd = xr.concat(drifting_gratings_cond_psd, dim=pd.Index(cond_presentation_id, name='condition_id'))
drifting_gratings_cond_psd = drifting_gratings_cond_psd.sel(condition_id=condition_id)


# In[18]:


freq_band = (20., 40.)
redo = True
while redo:
    freq_band, = get_parameters({'power_freq_band': freq_band}, parameters_dict, enter_parameters)

    cond_band_power = drifting_gratings_cond_psd.sel(frequency=slice(*freq_band)).integrate('frequency').rename(PSD='band power')

    temporal_frequency = cond_band_power.temporal_frequency.values
    nchannel = cond_band_power.dims['channel']

    _, axs = plt.subplots(nchannel, 1, squeeze=False, figsize=(5, 3 * nchannel))
    for c, ax in zip(cond_band_power.channel, axs.ravel()):
        cpower = cond_band_power.sel(channel=c).to_array()[0].T
        cpower.plot.imshow(ax=ax)
        ax.set_yticks(np.linspace(temporal_frequency[0], temporal_frequency[-1], temporal_frequency.size))
        ax.set_yticklabels(temporal_frequency)
    plt.tight_layout()

    fig_disp(f'drifting_grating_power_band_{freq_band[0]:g}-{freq_band[1]:g}')

    redo = whether_redo()


# In[19]:


redo = True
while redo:
    ap_knee, = get_parameters({'drifting_gratings_knee': True}, parameters_dict, enter_parameters=False)
    aperiodic_mode = 'knee' if ap_knee else 'fixed'

    tfreq_psd = drifting_gratings_cond_psd.mean(dim='orientation')
    for freq in drifting_gratings_cond_psd.temporal_frequency:
        print(f'\ntemporal_frequency: {freq: .1f} Hz')
        _, fig1, fig2 = plot_channel_psd(tfreq_psd.sel(temporal_frequency=freq), channel_id=group_id,
                             freq_range=200., plt_range=(0, 100.), figsize=(4, 3),
                             aperiodic_mode=aperiodic_mode, dB_threshold=3., max_n_peaks=10, plt_log=True)
        fig_disp({f'drifting_grating_tfreq_{freq:g}_psd': fig1, f'drifting_grating_tfreq_{freq:g}_fooof_channel_{group_id}': fig2})

    redo = whether_redo()


# In[20]:


redo = True
while redo:
    channel_ids, = get_parameters({'scaleogram_channels': list(map(int, lfp_array.channel))}, parameters_dict, enter_parameters)

    grating_aligned_lfp = align_trials(lfp_array, grating_ids, grating_times, window=(-0.5, grating_duration + 0.5))

    for freq in condition_id.temporal_frequency:
        print(f'\ntemporal_frequency: {freq: .1f} Hz')
        present_ids = np.concatenate([cond_presentation_id[i] for i in condition_id.sel(temporal_frequency=freq).values])
        sxx_avg = trial_averaged_spectrogram(grating_aligned_lfp.sel(presentation_id=present_ids), tseg=.25)
        _ = plot_channel_spectrogram(sxx_avg, channel_id=channel_ids, plt_range=(0, 100.), plt_log=True,
                                     clr_freq_range=(25., 40.), figsize=(5, 3),
                                     remove_aperiodic={'freq_range': 200., 'aperiodic_mode': aperiodic_mode})
        fig_disp(f'drifting_grating_tfreq_{freq:g}_scaleogram')

    redo = whether_redo()


# ## Save parameters in config

# In[21]:


with open(info_file, 'w') as f:
    json.dump(info, f, indent=4)


# In[ ]:




