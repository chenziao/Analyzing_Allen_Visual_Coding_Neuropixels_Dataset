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

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from utils import *

pd.set_option('display.max_columns', None)

with open('config.json') as f:
    config = json.load(f)


# In[3]:


# Cache directory path, it determines where downloaded data will be stored
manifest_path = os.path.join(config['cache_dir'], "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)


# In[4]:


VI_structure_acronyms = [x for x in cache.get_structure_acronyms() if x is not np.nan and 'VI' in x]
print('Visual cortex areas: ' + ', '.join(VI_structure_acronyms))


# In[5]:


sessions = cache.get_session_table()
pd.set_option('display.max_rows', 10)
display(sessions)


# ## Filter sessions data

# In[6]:


filter_dict = config['filter_dict']
ecephys_structure_acronym = filter_dict['ecephys_structure_acronyms']
filtered_sessions = sessions[
    np.logical_and.reduce([np.array([True if s in x else False for x in sessions[k]]) for k, s in filter_dict.items()])
]
pd.set_option('display.max_rows', None)
display(filtered_sessions)
print(filtered_sessions.index)


# In[7]:


session_id = int(input('Select session ID: '))


# In[8]:


session = cache.get_session_data(session_id)
channels = session.channels
channels = channels.loc[channels['structure_acronym'] == ecephys_structure_acronym].copy()


# ## Get units in area of interest

# In[9]:


# units in a session after quality filter
units = session.units
VI_units = units.loc[[i for i, x in units.iterrows() if x['ecephys_structure_acronym'] in VI_structure_acronyms]]
print(f'Total number of units in visual cortex: {len(VI_units):d}')


# In[10]:


probes = cache.get_probes()
probes = probes[probes['ecephys_session_id'] == session_id]

for i, x in probes.iterrows():
    print(f'probe id: {i:d}')
    areas = [s for s in x['ecephys_structure_acronyms'] if s in VI_structure_acronyms]
    n_units = [np.count_nonzero(VI_units['ecephys_structure_acronym'] == s) for s in areas]
    print('structure: ' + ', '.join(areas))
    print('number of units: ' + ', '.join(map(str, n_units)) + '\n')


# In[11]:


sel_units = VI_units[VI_units['ecephys_structure_acronym'] == ecephys_structure_acronym].copy()
if len(sel_units) == 0:
    raise SystemExit(f'No unit found in {ecephys_structure_acronym}.')


# In[12]:


# # Spike time data is missing for units filtered out in a session
# units = cache.get_units(amplitude_cutoff_maximum = np.inf,
#                         presence_ratio_minimum = -np.inf,
#                         isi_violations_maximum = np.inf)
# units = units[units['ecephys_session_id'] == session_id]

# VI_units_all = units.loc[[i for i, x in units.iterrows() if x['ecephys_structure_acronym'] in VI_structure_acronyms]]
# len(VI_units_all)


# ### Find center channel

# In[13]:


ccf_coords = ['anterior_posterior_ccf_coordinate', 'dorsal_ventral_ccf_coordinate', 'left_right_ccf_coordinate']
has_ccf = not np.any(channels[ccf_coords].isnull()) # whether ccf is missing
unit_has_ccf = not np.any(sel_units[ccf_coords].isnull()) # whether ccf is missing for units

if not has_ccf:
    # use probe coordinate to represent ccf
    channels[ccf_coords] = 0.
    channels['dorsal_ventral_ccf_coordinate'] = 3840. - channels['probe_vertical_position']

if not unit_has_ccf:
    # use channel coordinate to represent ccf
    sel_units[ccf_coords] = channels.loc[sel_units['peak_channel_id'], ccf_coords].values


# In[14]:


# units center
units_coord = sel_units[ccf_coords].values
units_coord_mean = units_coord.mean(axis=0)
center_unit_id = sel_units.index[np.argmin(np.sum((units_coord - units_coord_mean) ** 2, axis=1))]

channel_index = sel_units.loc[center_unit_id, 'probe_channel_number']
center_unit_probe_id = sel_units.loc[center_unit_id, 'probe_id']

channel = channels[(channels.probe_channel_number == channel_index) & (channels.probe_id == center_unit_probe_id)]
center_unit_channel_id = int(channel.index[0])

print('Channel of the unit near the center of units')
display(channel)


# In[15]:


# channels center
if has_ccf:
    channels_coord = channels[ccf_coords].values
    center_channel_id = int(channels.index[np.argmin(np.sum((channels_coord - units_coord_mean) ** 2, axis=1))])

    print('Chennel near the center of units')
    display(channels.loc[[center_channel_id]])
else:
    center_channel_id = None
    print('Channels CCF missing.')


# In[16]:


probe_id = channels.probe_id.unique()
print('Probes and channels in %s:' % ecephys_structure_acronym)
for i in probe_id:
    print(f'probe id {i:d}: {sum(channels.probe_id == i):d} channels')


# In[17]:


probe_id = int(input(f'Select probe: (center unit probe {center_unit_probe_id})') if probe_id.size > 1 else probe_id[0])
if not probes.loc[probe_id, 'has_lfp_data']:
    raise SystemExit(f'Probe {i:d} does not have LFP data.')
fs = probes.loc[probe_id, 'lfp_sampling_rate']


# ## Save probe information

# In[18]:


output_dir = config['output_dir']
session_dir = os.path.join(output_dir, f'session_{session_id:d}')
units_file = os.path.join(output_dir, f'session_{session_id:d}_{ecephys_structure_acronym:s}_units.csv')
info_file = os.path.join(output_dir, f'session_{session_id:d}_{ecephys_structure_acronym:s}.json')
if not os.path.isdir(session_dir):
    os.makedirs(session_dir)

if not os.path.isfile(units_file):
    sel_units.to_csv(units_file)

save_info = input('Save information file [y/n]?') if os.path.isfile(info_file) else 'y'
save_info = save_info and save_info[0].lower() == 'y'
if save_info:
    info = {
        'session_id': session_id,
        'ecephys_structure_acronym': ecephys_structure_acronym,
        'probe_id': probe_id,
        'center_channel_id': center_channel_id,
        'center_unit_channel_id': center_unit_channel_id,
        'fs': fs,
        'parameters': {}, # to be added in later analyses
        'has_ccf': has_ccf,
        'unit_has_ccf': unit_has_ccf,
    }
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)
else:
    with open(info_file) as f:
        info = json.load(f)


# ## Average channels in groups and save

# In[19]:


import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

whether_redo = redo_condition(True)
fig_disp = figure_display_function(config, session_id=session_id, ecephys_structure_acronym=ecephys_structure_acronym)


# In[20]:


redo = True
while redo:
    channel_group_by_ccf, = get_parameters({'channel_group_by_ccf': has_ccf}, info, enter_parameters=has_ccf)
    if channel_group_by_ccf:
        from sklearn.manifold import LocallyLinearEmbedding
        from sklearn.cluster import KMeans

        channel_coord = channels[ccf_coords].values
        # reduce dimension to one
        method = 'ltsa' # 'standard' 'hessian' 'ltsa' 'modified'
        lle = LocallyLinearEmbedding(n_components=1, n_neighbors=6, method=method)
        probe_coord = lle.fit_transform(channel_coord) # 1D coordinate of channels
        probe_coord = MinMaxScaler().fit_transform(probe_coord)

        # find number of clusters in the channels
        kmeans = KMeans(2, n_init=1)
        jumps = kmeans.fit_predict(np.diff(np.sort(probe_coord, axis=0), axis=0))
        n_group = np.sum(jumps == np.argmax(kmeans.cluster_centers_.ravel())) + 1

        # divide channels into groups
        kmeans = KMeans(n_group, n_init=1)
        channel_group = kmeans.fit_predict(probe_coord)
    else:
        from sklearn.decomposition import PCA
        
        channel_coord = channels[['probe_horizontal_position', 'probe_vertical_position']].to_numpy(dtype=float, copy=True)
        channel_coord[:, 1] = 3840. - channel_coord[:, 1]
        print('Channels range: {:.0f} microns'.format(channel_coord[:, 1].max() - channel_coord[:, 1].min()))
        n_group, = get_parameters({'n_channel_groups': 8}, info, enter_parameters=True)
        pca = PCA(n_components=1)
        probe_coord = pca.fit_transform(channel_coord)
        probe_coord = MinMaxScaler().fit_transform(probe_coord)

        # divide channels into groups
        sorted_probe_coord = np.sort(probe_coord.ravel())
        increments = np.diff(sorted_probe_coord)
        jumps = np.nonzero(increments > increments.mean())[0] + 1
        cuts = np.round(np.linspace(0, probe_coord.size, n_group + 1))
        cuts = jumps[pd.Index(jumps).get_indexer(cuts[1:-1], method='nearest')]
        cut_coord = (sorted_probe_coord[cuts - 1] + sorted_probe_coord[cuts]) / 2
        channel_group = np.searchsorted(cut_coord, probe_coord.ravel())

    # group center coordinate
    group_centers = np.zeros(n_group)
    group_count = np.zeros(n_group)
    for g, x in zip(channel_group, channel_coord[:, 1]):
        group_centers[g] += x
        group_count[g] += 1
    group_centers /= group_count

    # visualize channel groups
    plt.figure(figsize=(5, 4))
    for g, x in zip(channel_group, channel_coord[:, 1]):
        plt.plot(group_centers[g], x, 'b.', markersize=2)
    plt.xlabel('Group centers dorsal ventral coordinate')
    plt.ylabel('Channels dorsal ventral coordinate')
    fig_disp('channel_group_centers')

    redo = whether_redo()

if channel_group_by_ccf:
    info['n_channel_groups'] = int(n_group)


# In[21]:


probe_dir = os.path.join(session_dir, f'probe_{probe_id:d}')
if not os.path.isdir(probe_dir):
    os.makedirs(probe_dir)

filepath = os.path.join(probe_dir, f'{ecephys_structure_acronym:s}_lfp_channel_groups.nc')
save_lfp = input('LFP channel group data already exists. Overwrite [y/n]?') if not save_info and os.path.isfile(info_file) else 'y'
save_lfp = save_lfp and save_lfp[0].lower() == 'y'


# In[22]:


if save_lfp:
    # Compile groups of channels
    group_sort_id = np.argsort(group_centers)
    channels_group_id = np.zeros(channel_group.size, dtype=int)
    channels_in_groups = {}
    group_ccf_coord = np.zeros((n_group, 3))
    for i in range(n_group):
        idx = channel_group == group_sort_id[i]
        channels_group_id[idx] = i
        group_ccf_coord[i, :] = channels[idx][ccf_coords].values.mean(axis=0)
        channels_in_groups[i] = channels[idx].index.values
    channel_group_map = pd.DataFrame(channels_group_id, columns=['group_id'], index=channels.index)
    channel_group_map[ccf_coords] = group_ccf_coord[channels_group_id]

    # Load LFP given probe
    lfp_array = session.get_lfp(probe_id)
    lfp_array = lfp_array.sel(channel=np.unique(lfp_array.channel.sel(channel=channels.index, method='nearest')))
    for g, c in channels_in_groups.items():
        channels_in_groups[g] = np.array([x for x in c if x in lfp_array.channel])

    # Create group average LFP dataset
    channel_group_ids = pd.Index(np.arange(n_group), name='group_id')
    group_lfp = [lfp_array.sel(channel=channels_in_groups[i]).mean(dim='channel') for i in channel_group_ids]
    group_lfp = xr.concat(group_lfp, dim=channel_group_ids).to_dataset(name='LFP')
    print(group_lfp)

    group_lfp.to_netcdf(filepath) # save downsampled channels
    channel_group_map.to_csv(filepath.replace('.nc', '.csv'))


# ## Save LFP of particular channel

# In[23]:


# channel_id = center_channel_id
# lfp = session.get_lfp(probe_id).sel(channel=channel_id, method='nearest')
# filepath = os.path.join(probe_dir, f'lfp_channel_{channel_id:d}.nc')
# lfp.to_netcdf(filepath)


# ## Save parameters in config

# In[24]:


if save_lfp:
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)


# In[ ]:




