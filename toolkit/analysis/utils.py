import numpy as np
import xarray as xr

from typing import Sequence
from numpy.typing import ArrayLike, NDArray


def array_spacing(x : ArrayLike) -> float:
    """Get spacing of an evenly spaced array."""
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Input must be a 1D array.")
    if x.size < 2:
        raise ValueError("Input must have at least 2 elements.")
    return (x[-1] - x[0]) / (x.size - 1)


def get_bins(window : tuple[float, float] = (0., 1.), bin_width : float = 1.0, strict_window : bool = False) -> NDArray[float]:
    """Get bins for a given bin width within a window aligning one bin center at zero.
    Note that the actual window edges are rounded to the nearest bin width.
    
    Parameters
    ----------
    window : tuple[float, float]
        Window.
    bin_width : float
        Bin width.
    strict_window : bool
        Whether bin centers are strictly within the window.

    Returns
    -------
    bin_centers : NDArray[float]
        Bin centers.
    bin_edges : NDArray[float]
        Bin edges.
    """
    int_window = np.array(window) / bin_width
    if strict_window:
        int_window = int(np.ceil(int_window[0])), int(np.floor(int_window[1]))
    else:
        int_window = np.round(int_window).astype(int)
    bin_centers = np.arange(int_window[0], int_window[1] + 1) * bin_width
    bin_edges = np.append(bin_centers - bin_width / 2, bin_centers[-1] + bin_width / 2)
    return bin_centers, bin_edges


def stack_xarray_dims(
    da : xr.DataArray | xr.Dataset,
    dims : Sequence[str] = None,
    exclude_dims : Sequence[str] = None,
    stack_dim : str = 'sample',
    create_index : bool = False
) -> xr.DataArray | xr.Dataset:
    """Stack dimensions of a dataarray or dataset into a single dimension.
    
    Parameters
    ----------
    da : xr.DataArray | xr.Dataset
        Dataarray or dataset to stack dimensions.
    dims : Sequence[str]
        Dimensions to stack. Note that the order of dimensions in `dims`
        determines the stacked dimension in 'C' order.
    exclude_dims : Sequence[str]
        Dimensions to exclude from stacking. If specified, `dims` will be ignored
        and the remaining dimensions in the original order will be stacked.
    stack_dim : str
        Name of the new dimension. The new dimension will be the last dimension.
    create_index : bool
        If True, create a multi-index for the stacked dimension.
        If False, create one single (1-d) coordinate index for the new dimension
        and the original coordinates of the stacked dimension will be dropped.

    Returns
    -------
    da : xr.DataArray | xr.Dataset
        Dataarray or dataset with stacked dimensions.
    """
    if dims is None:
        dims = da.dims
    if exclude_dims is not None:
        dims = [d for d in da.dims if d not in exclude_dims]
    da = da.stack({stack_dim: dims}, create_index=create_index)
    if not create_index:  # make sure the new dimension is assigned coordinates
        stack_coord = da.coords[stack_dim]
        da = da.drop_vars(stack_coord.coords.keys())
        da = da.assign_coords({stack_dim: stack_coord.values})
    return da


def concat_stack_xarray_dims(
    das : Sequence[xr.DataArray | xr.Dataset],
    stack_dim : str = 'sample',
    reindex : bool = True,
    **kwargs
) -> xr.DataArray | xr.Dataset:
    """Stack dimensions of multiple dataarrays or datasets into a single dimension
    and concatenate them along the new dimension.
    
    Parameters
    ----------
    das : Sequence[xr.DataArray | xr.Dataset]
        Dataarrays or datasets to stack and concatenate.
    stack_dim : str
        Name of the new dimension.
    reindex : bool
        If True, reindex the new dimension to be consecutive integers.
    **kwargs : dict
        Keyword arguments for `stack_xarray_dims`.

    Returns
    -------
    da : xr.DataArray | xr.Dataset
        Dataarray or dataset with stacked dimensions and concatenated along the new dimension.
    """
    da = xr.concat([stack_xarray_dims(da, stack_dim=stack_dim, **kwargs) for da in das], dim=stack_dim)
    if reindex:
        stack_coord = da.coords[stack_dim]
        da = da.drop_vars(stack_coord.coords.keys())
        da = da.assign_coords({stack_dim: np.arange(stack_coord.size)})
    return da
