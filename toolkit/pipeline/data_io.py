from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

from ..paths import *


class SessionDirectory:
    def __init__(self, session_id : int, structure_acronym : str):
        """Get the cache directory for a given session and structure."""
        self._session_dir = PROCESSED_DATA_CACHE_DIR / structure_acronym / str(session_id)
        self.exist = self._session_dir.exists()
        self.cache = EcephysProjectCache.from_warehouse(manifest=ECEPHYS_MANIFEST_FILE)
        self.session = self.cache.get_session_data(session_id)

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

    def save_probe_info(self, probe_id : int, central_channels : dict[str, int]) -> None:
        probe_info = {
            'probe_id': probe_id,
            'central_channels': central_channels,
        }
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
        return xr.load_dataarray(self.csd)


