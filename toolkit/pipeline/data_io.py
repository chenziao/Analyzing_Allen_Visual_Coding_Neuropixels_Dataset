from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from ..utils.quantity_units import convert_unit, units_equal
from ..paths import *

if TYPE_CHECKING:
    from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession


class SessionDirectory:
    def __init__(self, session_id : int, structure_acronym : str, cache_lfp : bool = False):
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
        """
        self.session_id = session_id
        self._session_dir = PROCESSED_DATA_CACHE_DIR / structure_acronym / str(session_id)
        self.exist = self._session_dir.exists()
        self.cache : EcephysProjectCache = EcephysProjectCache.from_warehouse(manifest=ECEPHYS_MANIFEST_FILE)
        self.session : EcephysSession = self.cache.get_session_data(session_id)
        self._cache_lfp = cache_lfp
        self.lfp_cache : dict[int, xr.DataArray] = {}

    @property
    def session_dir(self) -> Path:
        if not self.exist:
            self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir


    @property
    def lfp_channels(self) -> Path:
        return self.session_dir / 'lfp_channels.csv'

    def save_lfp_channels(self, lfp_channels : pd.DataFrame) -> None:
        lfp_channels.to_csv(self.lfp_channels)

    def load_lfp_channels(self) -> pd.DataFrame:
        return pd.read_csv(self.lfp_channels, index_col='id')


    @property
    def probe_info(self) -> Path:
        return self.session_dir / 'probe_info.json'

    def save_probe_info(
        self,
        probe_id : int,
        central_channels : dict[str, int],
        csd_channels : list[int],
        csd_padding : tuple[int, int]
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
        """
        probe_info = dict(
            probe_id = int(probe_id),
            central_channels = {k: int(v) for k, v in central_channels.items()},
            csd_channels = [int(v) for v in csd_channels],
            csd_padding = csd_padding
        )
        with open(self.probe_info, 'w') as f:
            json.dump(probe_info, f, indent=4)

    def load_probe_info(self) -> dict:
        with open(self.probe_info, 'r') as f:
            return json.load(f)


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


    def get_lfp(self, probe_id : int) -> xr.DataArray:
        """Get LFP array from allensdk session cache and cache in memory.

        Parameters
        ----------
        probe_id : int
            Probe ID.

        Returns
        -------
        xr.DataArray
            LFP data. Unit: V
        """
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

    def load_lfp(self, probe_id, channel=None, time=None, unit='uV'):
        """Load LFP array from cache with optional selection of channels and time.

        Parameters
        ----------
        probe_id : int
            Probe ID.
        channel : list[int], optional
            Channels to load.
        time : list[int], optional
            Time to load.

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

    def probe_lfp_channels(self, probe_id : int) -> pd.DataFrame:
        """Load channels for a given probe.
        
        Parameters
        ----------
        probe_id : int
            Probe ID.
        """
        channels = self.session.channels.loc[self.get_lfp(probe_id).channel]
        # ensure sorted by vertical position
        return channels.sort_values('probe_vertical_position')


    @property
    def psd(self) -> Path:
        return self.session_dir / 'psd_channel_groups.nc'

    def save_psd(self, psd_das : dict[str, xr.DataArray], channel_groups : xr.DataArray) -> None:
        """Save PSD of stimuli into a single dataset.
        
        Parameters
        ----------
        psd_das : dict[str, xr.DataArray]
            Dictionary of {stimulus_name: PSD dataarray}.
        channel_groups : xr.DataArray
            Channel groups of the LFP data used for PSD calculation.
        """
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
        xr.Dataset
            PSD of stimuli. Unit: uV**2/Hz
        """
        return xr.load_dataset(self.psd)


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
        cond_psd_das = {stim: xr.load_dataset(f) for stim, f in zip(stimulus_names, cond_psd_files)}
        return cond_psd_das

