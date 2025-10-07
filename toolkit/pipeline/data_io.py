from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

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
        """
        self._session_dir = PROCESSED_DATA_CACHE_DIR / structure_acronym / str(session_id)
        self.exist = self._session_dir.exists()
        self.cache : EcephysProjectCache = EcephysProjectCache.from_warehouse(manifest=ECEPHYS_MANIFEST_FILE)
        self.session : EcephysSession = self.cache.get_session_data(session_id)
        self.cache_lfp = cache_lfp
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
            CSD data. Unit: μV/mm²
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
        if self.cache_lfp and probe_id in self.lfp_cache:
            return self.lfp_cache[probe_id]
        lfp_array = self.session.get_lfp(probe_id)
        probes = self.cache.get_probes()
        fs = probes.loc[probe_id, 'lfp_sampling_rate']
        lfp_array.attrs.update(fs=fs, unit='V')
        if self.cache_lfp:
            self.lfp_cache[probe_id] = lfp_array
        return lfp_array

    def clear_lfp_cache(self) -> None:
        self.lfp_cache.clear()

    def load_lfp(self, probe_id, channel=None, time=None):
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
            LFP data. Unit: V
        """
        lfp_array = self.get_lfp(probe_id)
        sel = {}
        if channel is not None:
            sel['channel'] = channel
        if time is not None:
            sel['time'] = time
        if sel:
            lfp_array = lfp_array.sel(**sel)
        return lfp_array

