from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import xarray as xr
from enum import Enum
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from ..utils.quantity_units import convert_unit, units_equal
from ..paths import *
from ..pipeline.global_settings import GLOBAL_SETTINGS

from typing import Any

if TYPE_CHECKING:
    from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


STRUCTURE_ACRONYM = GLOBAL_SETTINGS['ecephys_structure_acronym']


# Processed data directory

class SessionDirectory:
    def __init__(
        self,
        session_id : int,
        structure_acronym : str = STRUCTURE_ACRONYM,
        cache_lfp : bool = False
    ):
        """Get the cache directory for a given session and structure.

        Parameters
        ----------
        session_id : int
            Ecephys session ID.
        structure_acronym : str
            Acronym of the structure of interest.
        cache_lfp : bool, optional
            Whether to cache LFP arrays in memory. Each probe consumes 3.75 GB of memory. Default is False.

        Attributes
        ----------
        session_dir : Path
            Path to the session directory.
        cache : EcephysProjectCache
            EcephysProjectCache object.
        session : EcephysSession
            EcephysSession object.
        lfp_cache : dict[int, xr.DataArray]
            Dictionary of {probe_id: cached LFP arrays}.
        probe_id : int | None
            Probe ID. None if no probe info is available.
        has_lfp_data : bool
            Whether the probe has LFP data.
        """
        self.session_id = session_id
        self.structure_acronym = structure_acronym
        self._session_dir = PROCESSED_DATA_CACHE_DIR / structure_acronym / str(session_id)
        self.cache : EcephysProjectCache = EcephysProjectCache.from_warehouse(manifest=ECEPHYS_MANIFEST_FILE)
        self._session : EcephysSession | None = None
        self._cache_lfp = cache_lfp
        self.lfp_cache : dict[int, xr.DataArray] = {}
        self._probe_id = None
        self._has_lfp_data = None

    # Attributes
    @property
    def exist(self) -> bool:
        return self._session_dir.exists()

    @property
    def session_dir(self) -> Path:
        if not self.exist:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir

    @property
    def probe_id(self) -> int:
        if self._probe_id is None:
            self.load_probe_info()
        return self._probe_id

    @property
    def has_lfp_data(self) -> bool:
        if self._has_lfp_data is None and self.exist:
            try:
                self.load_probe_info()  # determine if has LFP data
            except FileNotFoundError:
                pass  # remains unknown
        return self._has_lfp_data

    @property
    def session(self) -> EcephysSession:
        if self._session is None:
            self._session : EcephysSession = self.cache.get_session_data(self.session_id)
        return self._session

    @property
    def genotype(self) -> str:
        """Return the abbreviated genotype: 'Pvalb', 'Sst', 'Vip', 'wt'"""
        from ..allen_helpers.units import get_genotype
        return get_genotype(self.session.full_genotype)

    # Probe info
    @property
    def probe_info(self) -> Path:
        return self.session_dir / 'probe_info.json'

    def save_probe_info(
        self,
        probe_id : int,
        central_channels : dict[str, int],
        csd_channels : list[int],
        csd_padding : tuple[int, int],
        **kwargs
    ) -> None:
        """Save probe information to a JSON file.
        
        Parameters
        ----------
        probe_id : int
            Probe ID.
        central_channels : dict[str, int]
            Layer structure acronym: ID of Central channels in the layer.
        csd_channels : list[int]
            Channels used for CSD calculation.
        csd_padding : tuple[int, int]
            Padding used for CSD calculation.
        kwargs : dict
            Additional information to save.
        """
        probe_info = dict(  # ensure dict type for JSON serialization
            has_lfp_data = True,
            probe_id = int(probe_id),
            central_channels = {k: int(v) for k, v in central_channels.items()},
            csd_channels = [int(v) for v in csd_channels],
            csd_padding = csd_padding
        ) | kwargs
        with open(self.probe_info, 'w') as f:
            json.dump(probe_info, f, indent=4)

    def save_null_probe_info(self) -> None:
        """Save null probe information (no LFP data) to a JSON file."""
        probe_info = dict(
            has_lfp_data = False,
            probe_id = None,
            central_channels = None,
            csd_channels = None,
            csd_padding = None
        )
        with open(self.probe_info, 'w') as f:
            json.dump(probe_info, f, indent=4)

    def load_probe_info(self) -> dict:
        if not self.probe_info.exists():
            raise FileNotFoundError(f"Probe info file {self.probe_info} not found")

        with open(self.probe_info, 'r') as f:
            probe_info = json.load(f)

        self._probe_id = probe_info['probe_id']
        self._has_lfp_data = bool(probe_info.get('has_lfp_data', self._probe_id is not None))
        return probe_info

    # LFP channels
    @property
    def lfp_channels(self) -> Path:
        return self.session_dir / 'lfp_channels.csv'

    def save_lfp_channels(self, lfp_channels : pd.DataFrame) -> None:
        lfp_channels.to_csv(self.lfp_channels)

    def load_lfp_channels(self) -> pd.DataFrame:
        return pd.read_csv(self.lfp_channels, index_col='id')

    # CSD
    @property
    def csd(self) -> Path:
        return self.session_dir / 'csd.nc'

    def save_csd(self, csd_array : xr.DataArray) -> None:
        csd_array.to_netcdf(self.csd)

    def load_csd(self) -> xr.DataArray:
        """Load CSD array from cache.
        
        Returns
        -------
        xr.DataArray
            CSD data. Unit: uV/mm**2
        """
        return xr.load_dataarray(self.csd)

    # LFP
    def get_lfp(self, probe_id : int | None = None) -> xr.DataArray:
        """Get LFP array from allensdk session cache and cache in memory.

        Parameters
        ----------
        probe_id : int | None
            Probe ID. If not provided, use the probe ID from the probe info file.

        Returns
        -------
        xr.DataArray
            LFP data. Unit: V
        """
        if probe_id is None:
            probe_id = self.probe_id

        if self._cache_lfp and probe_id in self.lfp_cache:
            return self.lfp_cache[probe_id]

        lfp_array = self.session.get_lfp(probe_id)
        probes = self.cache.get_probes()
        fs = probes.loc[probe_id, 'lfp_sampling_rate']
        lfp_array.attrs.update(fs=fs, unit='V')

        if self._cache_lfp:
            self.lfp_cache[probe_id] = lfp_array
        return lfp_array

    def clear_lfp_cache(self) -> None:
        self.lfp_cache.clear()

    def load_lfp(
        self,
        probe_id : int | None = None,
        channel : Any | None = None,
        time : Any | None = None,
        unit : str = 'uV'
    ) -> xr.DataArray:
        """Load LFP array from cache with optional selection of channels and time.

        Parameters
        ----------
        probe_id : int | None
            Probe ID. If not provided, use the probe ID from the probe info file.
        channel : xarray indexer, optional
            Channels to load.
        time : xarray indexer, optional
            Time to load.
        unit : str
            Desired unit of the LFP data. Default is 'uV'.

        Returns
        -------
        xr.DataArray
            LFP data. Unit: uV
        """
        lfp_array = self.get_lfp(probe_id)
        sel = {}
        if channel is not None:
            sel['channel'] = channel
        if time is not None:
            sel['time'] = time
        if sel:
            lfp_array = lfp_array.sel(**sel)
        src_unit = lfp_array.attrs['unit']
        if not units_equal(src_unit, unit):
            lfp_array = convert_unit(lfp_array, src_unit, unit, copy=False)
        return lfp_array

    def probe_lfp_channels(self, probe_id : int | None = None) -> pd.DataFrame:
        """Load channels for a given probe.
        
        Parameters
        ----------
        probe_id : int | None
            Probe ID. If not provided, use the probe ID from the probe info file.
        """
        from toolkit.allen_helpers.location import fill_missing_linear_channels
        # get probe channels
        all_channels = self.session.channels
        probe_channels = all_channels.loc[all_channels['probe_id'] == probe_id].sort_values('probe_channel_number')
        # fill possible missing lfp channels in the probe
        lfp_channels = self.get_lfp(probe_id).channel
        probe_channels = fill_missing_linear_channels(probe_channels, lfp_channels)
        # ensure sorted by vertical position
        channels = probe_channels.loc[lfp_channels].sort_values('probe_vertical_position')
        return channels

    # PSD
    @property
    def psd(self) -> Path:
        return self.session_dir / 'psd_channel_groups.nc'

    def save_psd(self, psd_das : dict[str, xr.DataArray] | xr.Dataset, channel_groups : xr.DataArray) -> None:
        """Save PSD of stimuli into a single dataset.
        
        Parameters
        ----------
        psd_das : dict[str, xr.DataArray]
            Dictionary of {stimulus_name: PSD dataarray}.
        channel_groups : xr.DataArray
            Channel groups of the LFP data used for PSD calculation.
        """
        if isinstance(psd_das, xr.Dataset):
            psd_das = dict(psd_das.data_vars)
        # Merge all PSDs into a dataset
        ds_attrs = next(iter(psd_das.values())).attrs
        ds_attrs = {key: ds_attrs[key] for key in ('fs', 'nfft', 'unit')}
        ds_attrs['session_id'] = self.session_id
        psd_ds = xr.Dataset(psd_das | dict(channel_groups=channel_groups), attrs=ds_attrs)
        psd_ds.to_netcdf(self.psd)

    def load_psd(self) -> xr.Dataset:
        """Load PSD of stimuli from a single dataset.
        
        Returns
        -------
        psd_ds : xr.Dataset
            PSD of stimuli. Unit: uV**2/Hz
        channel_groups : xr.DataArray
            Channel groups of the LFP data used for PSD calculation.
        """
        psd_ds = xr.load_dataset(self.psd)
        channel_groups = psd_ds.data_vars['channel_groups']
        psd_ds = psd_ds.drop_vars('channel_groups')
        return psd_ds, channel_groups

    # Conditions PSD
    def conditions_psd(self, stimulus_name : str) -> Path:
        return self.session_dir / f'{stimulus_name}_conditions_psd.nc'

    def save_conditions_psd(self, cond_psd_das : dict[str, xr.DataArray]) -> None:
        """Save conditions PSD of stimuli into separate data files.

        Parameters
        ----------
        cond_psd_das : dict[str, xr.DataArray]
            Dictionary of {stimulus_name: conditions PSD dataarray}.
        """
        for stim, cond_psd in cond_psd_das.items():
            cond_psd_file = self.conditions_psd(stim)
            cond_psd.attrs.update(session_id=self.session_id, stimulus_name=stim)
            cond_psd.to_netcdf(cond_psd_file)

    def load_conditions_psd(self) -> dict[str, xr.DataArray]:
        """Load conditions PSD of stimuli from separate data files.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary of {stimulus_name: conditions PSD dataarray}. Unit: uV**2/Hz
        """
        cond_psd_files = list(self.session_dir.glob("*_conditions_psd.nc"))
        stimulus_names = [f.stem.removesuffix('_conditions_psd') for f in cond_psd_files]
        cond_psd_das = {stim: xr.load_dataarray(f) for stim, f in zip(stimulus_names, cond_psd_files)}
        return cond_psd_das

    # Wave bands
    @property
    def wave_bands(self) -> Path:
        return self.session_dir / 'wave_bands.nc'

    def save_wave_bands(self, wave_bands : xr.Dataset) -> None:
        """Save detected frequency bands from FOOOF analysis.
        
        Parameters
        ----------
        bands : xr.Dataset
            Dataset containing:
            - bands : xr.DataArray
                Detected frequency bands from FOOOF analysis.
                Dimensions: stimulus, layer, wave_band, bound
            - wave_band_limit : xr.DataArray
                Wave band limit.
                Dimensions: wave_band
            - wave_band_width_limit : xr.DataArray
                Wave band width limit.
                Dimensions: wave_band
        """
        wave_bands.to_netcdf(self.wave_bands)

    def load_wave_bands(self) -> xr.Dataset:
        """Load detected frequency bands from FOOOF analysis."""
        return xr.load_dataset(self.wave_bands)

    # Band power in conditions
    def condition_band_power(self, stimulus_name : str, wave_band : str = 'beta') -> Path:
        return self.session_dir / f'{stimulus_name}_condition_{wave_band}_power.nc'

    def save_condition_band_power(
        self, cond_band_power_das : dict[str, xr.DataArray],
        wave_band : str = 'beta'
    ) -> None:
        """Save band power in drifting grating conditions into separate data files.

        Parameters
        ----------
        condition_band_power_das : dict[str, xr.DataArray]
            Dictionary of {stimulus_name: condition band power dataarray}.
            Dimensions: layer, *condition_types (e.g. orientation, temporal_frequency, contrast)
        wave_band : str
            Wave band of which the power is calculated.
        """
        for stim, cond_band_power in cond_band_power_das.items():
            cond_band_power.to_netcdf(self.condition_band_power(stim, wave_band))

    def load_condition_band_power(self, wave_band : str = 'beta', session_type : str | None = None) -> dict[str, xr.DataArray]:
        """Load band power in drifting grating conditions from separate data files.
        
        Parameters
        ----------
        wave_band : str
            Wave band of which the power is calculated.
        session_type : str | None
            Session type. If None, use the session type from the session cache.

        Returns
        -------
        dict[str, xr.DataArray]
            Dictionary of {stimulus_name: condition band power dataarray}.
            Dimensions: layer, *condition_types (e.g. orientation, temporal_frequency, contrast)
        """
        from ..allen_helpers.stimuli import STIMULUS_CATEGORIES
        if session_type is None:
            session_type = self.session.session_type
        drifting_gratings_stimuli = STIMULUS_CATEGORIES[session_type]['drifting_gratings']
        cond_band_power_das = {}
        for stim in drifting_gratings_stimuli:
            cond_band_power_file = self.condition_band_power(stim, wave_band)
            if cond_band_power_file.exists():
                cond_band_power_das[stim] = xr.load_dataarray(cond_band_power_file)
        return cond_band_power_das

    # orientation with max power
    def preferred_orientations(self, wave_band : str = 'beta') -> Path:
        return self.session_dir / f'preferred_orientations_{wave_band}.nc'

    def save_preferred_orientations(self, preferred_orientations : xr.DataArray, wave_band : str = 'beta') -> None:
        preferred_orientations.to_netcdf(self.preferred_orientations(wave_band))

    def load_preferred_orientations(self, wave_band : str = 'beta') -> xr.DataArray:
        return xr.load_dataarray(self.preferred_orientations(wave_band))

    # Stimulus trial averaged CSD
    def stimulus_csd(self, stimulus_name : str) -> Path:
        return self.session_dir / f'{stimulus_name}_csd.nc'

    def save_stimulus_csd(self, csd_dss : dict[str, xr.Dataset]) -> None:
        """Save stimulus trial averaged CSDs into separate data files.

        Parameters
        ----------
        csd_dss : dict[str, xr.Dataset]
            Dictionary of {stimulus_name: stimulus CSD dataset}.
        """
        for stim, ds in csd_dss.items():
            ds.to_netcdf(self.stimulus_csd(stim))

    def load_stimulus_csd(self) -> dict[str, xr.Dataset]:
        """Load stimulus trial averaged CSDs from separate data files.
        
        Returns
        -------
        csd_dss : dict[str, xr.Dataset]
            Dictionary of {stimulus_name: stimulus CSD dataset}.
        """
        csd_files = list(self.session_dir.glob("*_csd.nc"))
        stimulus_names = [f.stem.removesuffix('_csd') for f in csd_files]
        csd_dss = {stim: xr.load_dataset(f) for stim, f in zip(stimulus_names, csd_files)}
        return csd_dss

    # Stimulus LFP power
    def stimulus_lfp_power(self, stimulus_name : str) -> Path:
        return self.session_dir / f'{stimulus_name}_lfp_power.nc'

    def save_stimulus_lfp_power(self, lfp_power_dss : dict[str, xr.Dataset]) -> None:
        for stim, ds in lfp_power_dss.items():
            ds.to_netcdf(self.stimulus_lfp_power(stim))

    def load_stimulus_lfp_power(self) -> dict[str, xr.Dataset]:
        lfp_power_files = list(self.session_dir.glob("*_lfp_power.nc"))
        stimulus_names = [f.stem.removesuffix('_lfp_power') for f in lfp_power_files]
        lfp_power_dss = {stim: xr.load_dataset(f) for stim, f in zip(stimulus_names, lfp_power_files)}
        return lfp_power_dss

    # Units information
    @property
    def units_info(self) -> Path:
        return self.session_dir / 'units_info.csv'

    def save_units_info(self, units_info : pd.DataFrame) -> None:
        units_info.to_csv(self.units_info)

    def load_units_info(self) -> pd.DataFrame:
        return pd.read_csv(self.units_info, index_col='unit_id')

    # Units spike rate
    def units_spike_rate(self, stimulus_name : str) -> Path:
        return self.session_dir / f'{stimulus_name}_unit_spike_rate.nc'

    def save_units_spike_rate(self, units_spk_rate_dss : dict[str, xr.DataArray]) -> None:
        for stim, units_spk_rate in units_spk_rate_dss.items():
            units_spk_rate.to_netcdf(self.units_spike_rate(stim))

    def load_units_spike_rate(self) -> dict[str, xr.DataArray]:
        units_spk_rate_files = list(self.session_dir.glob("*_unit_spike_rate.nc"))
        stimulus_names = [f.stem.removesuffix('_unit_spike_rate') for f in units_spk_rate_files]
        units_spk_rate_dss = {stim: xr.load_dataarray(f) for stim, f in zip(stimulus_names, units_spk_rate_files)}
        return units_spk_rate_dss  


# File paths for non-session-specific files

class Files:
    """Dictionary of output file paths. Accessible as attributes or dictionary items."""
    _files = dict(
        session_selection=RESULTS_DIR / 'session_selection.csv',
        optotagged_sessions=RESULTS_DIR / 'optotagged_sessions.csv',
        bands_of_interest=RESULTS_DIR / 'bands_of_interest.nc',
        all_units_info=RESULTS_DIR / 'all_units_info.csv',
    )

    _index_col = dict(
        session_selection='id',
        optotagged_sessions='id',
        all_units_info='unit_id',
        layer_portion_boundary='layer',
    )

    _xarray_datatype = dict(
        bands_of_interest='dataarray',
        average_wave_bands='dataset',
    )

    # Helper methods
    def __getitem__(self, name: str) -> Path:
        """Get file path as dictionary item."""
        file = self._files.get(name, None)
        if file is None:
            raise AttributeError(f"File '{name}' is not defined.")
        return file

    def __getattr__(self, name: str) -> Path:
        """Get file path as attribute."""
        return self[name]

    def load(self, name: str, *args, **kwargs) -> pd.DataFrame | xr.Dataset:
        """Load data from a file.
        
        Parameters
        ----------
        name : str
            Name of the file to load.
        *args, **kwargs :
            Arguments to pass to the function for the file path with arguments.

        Returns
        -------
        data : pd.DataFrame | xr.Dataset
            Data from the file.
        """
        if len(args) > 0 or len(kwargs) > 0:
            file = getattr(self, name)(*args, **kwargs)
        else:
            try:
                file = self[name]
            except AttributeError:  # not in dictionary, try to get as method
                file = getattr(self, name)()  # using default arguments
        if not file.exists():
            raise FileNotFoundError(f"File '{file}' does not exist.")
        file_type = file.suffix.lstrip('.')
        match file_type:
            case 'csv':
                index_col=self._index_col.get(name)
                data =  pd.read_csv(file, index_col=index_col)
            case 'nc':
                datatype = self._xarray_datatype.get(name, 'dataset')
                data = getattr(xr, f'load_{datatype}')(file)
            case _:
                raise ValueError(f"Unsupported file type: '{file_type}'")
        return data

    def _data_dir(self, structure_acronym: str | None = None) -> Path:
        if structure_acronym is None:
            structure_acronym = STRUCTURE_ACRONYM
        return PROCESSED_DATA_CACHE_DIR / structure_acronym

    # File paths with arguments
    def layer_portion_boundary(self, structure_acronym: str | None = None) -> Path:
        return self._data_dir(structure_acronym) / 'layer_portion_boundary.csv'

    def wave_bands(self, session_type : str) -> Path:
        return RESULTS_DIR / f'wave_bands_{session_type}.csv'

    def average_wave_bands(self,
        session_type: str, session_set: str,
        structure_acronym: str | None = None
    ) -> Path:
        return self._data_dir(structure_acronym) / f'average_wave_bands_{session_type}_{session_set}.nc'

FILES = Files()  # Create instance of Files


# Output directory

class SessionSet(Enum):
    ALL = 'all'
    TEST = 'test'
    SELECTED = 'selected'
    UNSELECTED = 'unselected'
    OPTOTAG = 'optotag'
    SELECTED_OPTOTAG = 'selected_optotag'
    CUSTOM = 'custom'


def get_sessions(session_set : SessionSet | str | list[int]) -> tuple[list[int], SessionSet]:
    """Get the sessions from the session set.

    Parameters
    ----------
    session_set: SessionSet | str | list[int]
        The session set to get the sessions from. Can be a SessionSet enum, a string, or a list of session IDs.

    Returns
    -------
    sessions : list[int]
        The list of sessions from the session set.
    session_set : SessionSet
        The session set.
    """
    if isinstance(session_set, list):
        sessions = session_set
        session_set = SessionSet.CUSTOM
        return sessions, session_set

    if isinstance(session_set, str):
        session_set = SessionSet(session_set.lower())

    match session_set:
        case SessionSet.ALL:
            # Get all sessions available in the database
            from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
            cache = EcephysProjectCache.from_warehouse(manifest=paths.ECEPHYS_MANIFEST_FILE)
            sessions = cache.get_session_table().index.to_list()

        case SessionSet.TEST:
            # Get test sessions from the sessions file
            with open(paths.SESSIONS_FILE, 'r') as f:
                sessions_config = json.load(f)
            sessions = sessions_config.get('test', [])
            if not sessions:
                raise ValueError("No test sessions found in the sessions file")

        case SessionSet.SELECTED:
            # Get selected sessions from the session selection file
            sessions_df = get_session_selection()
            sessions = sessions_df.index[sessions_df['selected']].to_list()

        case SessionSet.UNSELECTED:
            # Get unselected sessions from the session selection file
            sessions_df = get_session_selection()
            # Unselected sessions that have valid data
            idx = sessions_df['has_structure'] & sessions_df['has_lfp_data'] & ~sessions_df['selected'] 
            sessions = sessions_df.index[idx].to_list()

        case SessionSet.OPTOTAG:
            # Get optotagged sessions from the optotagged sessions file
            optotagged_sessions_df = get_optotagged_sessions()
            sessions = optotagged_sessions_df.index.to_list()

        case SessionSet.SELECTED_OPTOTAG:
            # Get selected optotagged sessions from the optotagged sessions file
            optotagged_sessions_df = get_optotagged_sessions()
            sessions = optotagged_sessions_df.index[optotagged_sessions_df['selected']].to_list()

        case SessionSet.CUSTOM:
            raise ValueError("Custom sessions need to be provided as a list of session IDs")

    return sessions, session_set


def get_session_selection(structure_acronym : str = STRUCTURE_ACRONYM) -> pd.DataFrame:
    """Get the session selection dataframe from the output directory.
    
    Parameters
    ----------
    structure_acronym : str
        The structure acronym to get the session selection from.

    Returns
    -------
    pd.DataFrame
        Session selection dataframe. Create one if not exists.
    """
    file = FILES.session_selection
    if file.exists():
        sessions_df = FILES.load('session_selection')
    else:
        file.parent.mkdir(parents=True, exist_ok=True)
        sessions_df = initialize_session_selection(structure_acronym)
        sessions_df.to_csv(file)
    return sessions_df


def initialize_session_selection(structure_acronym : str = STRUCTURE_ACRONYM) -> pd.DataFrame:
    cache = EcephysProjectCache.from_warehouse(manifest=ECEPHYS_MANIFEST_FILE)
    sessions = cache.get_session_table()

    sessions_df = sessions[['session_type', 'full_genotype', 'unit_count']].copy()
    has_structure = []
    has_lfp_data = []

    for session_id in sessions.index:
        has_structure.append(structure_acronym in sessions.loc[session_id, 'ecephys_structure_acronyms'])
        session_dir = SessionDirectory(session_id, structure_acronym)
        has_lfp_data.append(session_dir.has_lfp_data if session_dir.exist else None)

    sessions_df['has_structure'] = has_structure
    sessions_df['has_lfp_data'] = has_lfp_data
    sessions_df['selected'] = sessions_df['has_structure'] & sessions_df['has_lfp_data']
    return sessions_df


def get_existing_sessions(
    session_set : SessionSet | str | list[int],
    structure_acronym : str = STRUCTURE_ACRONYM
) -> tuple[list[int], list[int]]:
    """Get the existing sessions in the data cache directory.
    
    Parameters
    ----------
    session_set : SessionSet | str | list[int]
        The session set to get the sessions from.
    structure_acronym : str
        The structure acronym to get the sessions from.

    Returns
    -------
    session_list : list[int]
        The list of existing sessions.
    missing_sessions : list[int]
        The list of missing sessions.
    """
    data_dir = PROCESSED_DATA_CACHE_DIR / structure_acronym
    session_list = []
    missing_sessions = []
    for s in get_sessions(session_set)[0]:
        f = data_dir / str(s)
        if f.is_dir():
            session_list.append(s)
        else:
            missing_sessions.append(s)

    if missing_sessions:
        print("Sessions missing from the data cache directory:")
        print('\n'.join(map(str, missing_sessions)))
    return session_list, missing_sessions


def save_optotagged_sessions(
    optotagged_sessions : pd.DataFrame,
    structure_acronym : str = STRUCTURE_ACRONYM,
    session_selection : pd.DataFrame | None = None
) -> None:
    from toolkit.utils.misc import pd_merge_differences
    if session_selection is None:
        session_selection = get_session_selection(structure_acronym)
    optotagged_sessions = pd_merge_differences(session_selection, optotagged_sessions,
        left_index=True, right_index=True, how='right')
    optotagged_sessions.index.name = session_selection.index.name
    file = FILES.optotagged_sessions
    file.parent.mkdir(parents=True, exist_ok=True)
    optotagged_sessions.to_csv(file)


def get_optotagged_sessions() -> pd.DataFrame:
    try:
        optotagged_sessions = FILES.load('optotagged_sessions')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\nPlease run `notebooks/compile_optotag.ipynb` to create it.")
    return optotagged_sessions


def format_for_path(path : str) -> str:
    """Format a string to be valid for a path"""
    return re.sub(r'[\\/:*?"<>|]', '_', path)
