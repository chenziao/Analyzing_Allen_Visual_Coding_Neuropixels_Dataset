from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
import numpy as np
import pandas as pd

from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes


CCF_COORDS = [
    'anterior_posterior_ccf_coordinate',
    'dorsal_ventral_ccf_coordinate',
    'left_right_ccf_coordinate'
]

CCF_AXES = dict(zip(['AP', 'DV', 'ML'], CCF_COORDS))  # map: axis acronym -> CCF coordinate name


def fill_missing_linear_channels(channels_df : pd.DataFrame, channels : NDArray[int]) -> pd.DataFrame:
    """Fill missing channels with neighbor channels values and linear interpolate positional values.
    
    Parameters
    ----------
    channels_df : pd.DataFrame
        DataFrame of existing channels from a single probe with Allen SDK channels table format.
    channels : NDArray[int]
        Array of a linear array of channels to be filled (matches lfp xarray 'channel').

    Returns
    -------
    pd.DataFrame
        DataFrame with all channels filled.
    """
    if channels_df['probe_id'].nunique() > 1:
        raise ValueError("Channels must be from the same probe.")

    # Create missing channels with neighbor channels values
    missing_channels = np.setdiff1d(channels, channels_df.index)
    if len(missing_channels) == 0:
        return channels_df.copy()
    neighbor_channels = channels_df.index.get_indexer(missing_channels, method='nearest')
    filled_channels = channels_df.iloc[neighbor_channels].copy()
    filled_channels.index = missing_channels

    # Interpolate positional values for missing channels
    int_cols = ['probe_channel_number', 'probe_horizontal_position', 'probe_vertical_position']
    interpolated_cols = int_cols + CCF_COORDS
    filled_channels[interpolated_cols] = np.nan  # set to nan to be interpolated
    filled_channels_df = pd.concat([channels_df, filled_channels]).sort_index()  # combine with existing channels
    filled_channels_df[interpolated_cols] = filled_channels_df[interpolated_cols].interpolate(method='linear')
    # Round all values following allen standard
    filled_channels_df.loc[missing_channels] = np.round(filled_channels_df.loc[missing_channels])
    filled_channels_df[int_cols] = filled_channels_df[int_cols].astype(int)
    # set index name to be the same
    filled_channels_df.index.name = channels_df.index.name
    return filled_channels_df


def get_lfp_channel_positions(cache_dir : str | Path, session_id : int, probe_id : int) -> pd.DataFrame:
    """
    Return electrode positions for the LFP channels of a given probe,
    aligned to the channel order in session.get_lfp(probe_id).

    Parameters
    ----------
    cache_dir : str | Path
        Base cache directory where NWB files are stored.
    session_id : int
        Ecephys session ID.
    probe_id : int
        Probe ID.

    Returns
    -------
    DataFrame
        Indexed by 'channel' (matches lfp xarray 'channel').
        Columns include ['local_index','probe_vertical_position',
                         'probe_horizontal_position','x','y','z',
                         'group_name','location'].

    Notes
    -----
    The 'x', 'y', 'z' coordinates in NWB file correspond to the CCF coordinates
    'anterior_posterior', 'dorsal_ventral', 'left_right', respectively.
    However, the 'z' coordinate is mistakenly replicated from the 'y' coordinate.
    Use with caution.
    """
    lfp_path = Path(cache_dir) / f"session_{session_id}" / f"probe_{probe_id}_lfp.nwb"
    if not lfp_path.exists():
        raise FileNotFoundError(f"LFP file not found at {lfp_path}")

    from pynwb import NWBHDF5IO

    with NWBHDF5IO(lfp_path, 'r') as io:
        nwb = io.read()
        # Grab the LFP ElectricalSeries
        es = nwb.acquisition[f"probe_{probe_id}_lfp_data"]
        # Indices of the LFP channels into the file's electrodes table
        lfp_elec_idx = np.asarray(es.electrodes.data)
        # Subset electrodes table in the given order
        elec_df = nwb.electrodes.to_dataframe().iloc[lfp_elec_idx].copy()

    # Rename 'id' to 'channel' and set as index
    elec_df = elec_df.reset_index().rename(columns={'id': 'channel'}).set_index('channel')
    return elec_df


class StructureFinder:
    # CCF axes: (AP, DV, ML) (per Allen docs). Directions: posterior, inferior, right
    AXES = dict(zip(CCF_AXES, range(3)))

    def __init__(self, cache_dir : str | Path, structure_acronym : str = 'VISp', voxel : int = 25):
        """
        Find the structure in the CCF annotation volume.

        Parameters
        ----------
        cache_dir : str | Path
            Reference space cache directory where structure tree and CCF annotation are stored.
        structure_acronym : str
            Cortical structure acronym (e.g. 'VISp').
        voxel : int
            Voxel size (µm, use 10, 25, 50...) for CCF annotation resolution.
        """
        from allensdk.core.reference_space_cache import ReferenceSpaceCache

        self.cache_dir = Path(cache_dir)
        self.reference_space_cache_dir = self.cache_dir / "annotation" / "ccf_2017"
        self.structure_acronym = structure_acronym
        self.voxel = int(voxel)
    
        # Load structure tree. Manifest will be cached locally
        self.reference_space_cache_dir.mkdir(exist_ok=True, parents=True)
        self.rsc = ReferenceSpaceCache(
            resolution=self.voxel,
            reference_space_key="annotation/ccf_2017",
            manifest=str(self.cache_dir / "manifest.json")
        )
        self.tree = self.rsc.get_structure_tree()

        # Get cortical structure
        self.structure = self.tree.get_structures_by_acronym([self.structure_acronym])[0]

        # Get layer structures
        layer_structures = self.tree.children([self.structure['id']])[0]
        self.layer_structures = {s['acronym']: s for s in layer_structures}  # index by acronym
        # map to simpler acronyms
        self.layer_acronym_map = {s: s.removeprefix(structure_acronym) for s in self.layer_structures}

        # CCF annotation
        self._annotation : NDArray[int] | None = None

    @property
    def annotation(self) -> NDArray[int]:
        if self._annotation is None:
            self.get_ccf_annotation()
        return self._annotation

    def get_ccf_annotation(self) -> NDArray[int]:
        """Get CCF annotation volume array (AP, DV, ML). Saved to cache directory."""
        annot_file = f"annotation_{self.voxel:d}.npy"
        annot_path = self.reference_space_cache_dir / annot_file
        if not annot_path.exists():
            self._annotation, meta = self.rsc.get_annotation_volume()
            # Note: The axes order in the meta data 'space' was wrong.
            np.save(annot_path, self._annotation)
        else:
            self._annotation = np.load(annot_path)
        return self._annotation

    def get_hemisphere_annotation(self, hemisphere : str = 'left') -> NDArray[int]:
        """Get hemisphere annotation"""
        ML_AXIS = self.AXES['ML']
        ml_size = self.annotation.shape[ML_AXIS]
        ml_mid = ml_size // 2  # midline index

        ml_slice = [slice(None)] * 3
        ml_slice[ML_AXIS] = slice(0, ml_mid) if hemisphere == 'left' else slice(ml_mid, ml_size)
        hemi_annot = self.annotation[tuple(ml_slice)]
        return hemi_annot

    def plot_layer_structures_scatter(
        self,
        scatter_density : float = 5000.,
        hemisphere : str = 'left',
        colormap : str = 'rainbow_r',
        ax : Axes | None = None
    ) -> Axes:
        """Plot layer structures scatter plot.

        Parameters
        ----------
        scatter_density : float
            Scatter density (voxels per mm^3) to speed up plotting.
        hemisphere : str
            Hemisphere ('left' or 'right', or '' for both).
        colormap : str
            Colormap name for different layers.
        ax : Axes
            Axes object to plot on.

        Returns
        -------
        Axes
            The axes object of the plot.
        """
        import matplotlib.pyplot as plt
        from toolkit.plots.utils import set_equal_3d_scaling

        AXES = ('ML', 'AP', 'DV')
        axes = list(map(self.AXES.get, AXES))
        scatter_prop = scatter_density * (self.voxel / 1000.) ** 3  # proportion of voxels to plot

        # Create colors dictionary using colormap
        jet = plt.get_cmap(colormap, len(self.layer_structures))
        colors = {acr: jet(i) for i, acr in enumerate(self.layer_structures)}

        # Get annotation
        annot = self.get_hemisphere_annotation(hemisphere) if hemisphere else self.annotation

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw=dict(projection='3d'))

        for acr, struct in self.layer_structures.items():
            mask = (annot == struct['id'])
            coords = np.argwhere(mask)
            if coords.size == 0:
                continue

            # Plot only a subsample for speed
            n_voxels = coords.shape[0]
            n_samples = min(int(n_voxels * scatter_prop), n_voxels)
            idx = np.random.choice(n_voxels, size=n_samples, replace=False)

            # Convert voxel indices to microns
            scatters = coords[idx].T * self.voxel
            ax.plot(*scatters[axes], color=colors[acr],
                linestyle='none', marker='.', markersize=2, label=acr, alpha=0.5)

        ax.set_xlabel(f'{AXES[0]} (µm)')
        ax.set_ylabel(f'{AXES[1]} (µm)')
        ax.set_zlabel(f'{AXES[2]} (µm)')
        ax.legend()
        ax.set_title(f'{hemisphere.capitalize()} {self.structure_acronym} Cortex'.title())

        data_lim = set_equal_3d_scaling(ax)

        ax.set_ylim3d(data_lim[1][::-1])  # flip DV
        ax.set_zlim3d(data_lim[2][::-1])  # flip ML
        return ax

    def world_um_to_idx(self, coords : ArrayLike) -> NDArray[int]:
        """Convert world coordinates (µm) to voxel indices.
        Array shape: (locations, coordinates (AP, DV, ML)). Clips to valid bounds.
        """
        idx = np.round(np.asarray(coords) / self.voxel).astype(int)
        idx = np.clip(idx, 0, np.array(self.annotation.shape) - 1)
        return idx

    def get_structure_array(self, coords : ArrayLike) -> tuple[list[str], list[bool]]:
        """Get array of layer structures from world coordinates.

        Parameters
        ----------
        coords : ArrayLike
            World coordinates (µm). Array shape: (locations, coordinates (AP, DV, ML)).

        Returns
        -------
        layer_acronym : list[str]
            list of layer structure acronyms (empty string if not a structure)
            with parent structure acronym removed.
        inside_structure : list[bool]
            A boolean array indicating whether the structure is inside the cortical structure.
        """
        ccf_idx = self.world_um_to_idx(coords).T  # (coordinates, locations)
        structure_array = self.tree.get_structures_by_id(self.annotation[tuple(ccf_idx)])
        layer_acronym = [s['acronym'] if s else '' for s in structure_array]

        structure_id = self.structure['id']
        inside_structure = []
        for i, s in enumerate(structure_array):
            parent_id = -1 if s is None else self.tree.parents([s['id']])[0]['id']
            inside = parent_id == structure_id
            inside_structure.append(inside)
            if inside:
                layer_acronym[i] = self.layer_acronym_map[layer_acronym[i]]
        return layer_acronym, inside_structure


def central_channel_in_structure(layer_acronyms : ArrayLike) -> dict[str, int]:
    """Find the central channel in the structure given an array of layer acronyms for each channel."""
    layer_acronyms = np.asarray(layer_acronyms, dtype=str)
    # get unique layer structures preserving order
    layer_structures = list(dict.fromkeys(layer_acronyms, 0))
    central_channel_idx = {}
    for layer_structure in layer_structures:
        idx = np.nonzero(layer_acronyms == layer_structure)[0]
        # get middle index (lower index if even number of channels)
        central_channel_idx[layer_structure] = idx[(len(idx) - 1) // 2]
    return central_channel_idx

