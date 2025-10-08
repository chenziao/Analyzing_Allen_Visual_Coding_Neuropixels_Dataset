"""
Unit conversion utilities.

Add converter for each unit pair in CONVERSION_TABLE.

Usage:
>>> from toolkit.utils.units import convert_unit
>>> x = xr.DataArray(np.full(5, 1e-6), coords={'time': np.arange(5.)}, attrs={'unit': 'V'})
>>> x = convert_unit(x, 'V', 'uV', copy=False)
>>> print(x)
<xarray.DataArray (time: 5)>
array([1., 1., 1., 1., 1.])
Coordinates:
  * time     (time) float64 0.0 1.0 2.0 3.0 4.0
Attributes:
    unit:     uV
"""

import quantities as pq
from quantities import UnitQuantity, markup
from contextlib import contextmanager
import warnings

from numpy.typing import NDArray
from pandas import DataFrame, Series
from xarray import DataArray

Array = NDArray | DataArray | DataFrame | Series


# Set default to use ASCII units
markup.config.use_unicode = False

@contextmanager
def unicode_units(use_unicode=True):
    """Context manager to temporarily set the use_unicode flag for quantities."""
    prev = markup.config.use_unicode
    try:
        markup.config.use_unicode = use_unicode
        yield
    finally:
        markup.config.use_unicode = prev


def str_to_unit(unit : str) -> UnitQuantity:
    """Convert a unit string to a quantities unit object."""
    return pq.unit_registry[unit]


def unit_to_str(quantity : UnitQuantity, use_unicode : bool = False) -> str:
    """Convert a quantities unit object to a string."""
    with unicode_units(use_unicode):
        return str(quantity.dimensionality)


def display_unit(unit : str) -> str:
    """Display the unit string with unicode."""
    return unit_to_str(str_to_unit(unit), use_unicode=True)


def units_equal(unit1 : str, unit2 : str) -> bool:
    """Check if two unit strings are equal."""
    with unicode_units(False):
        return str_to_unit(unit1) == str_to_unit(unit2)


def conversion_factor(src_unit : str, dst_unit : str) -> float:
    """Get the conversion factor between two units."""
    quantity = str_to_unit(src_unit) / str_to_unit(dst_unit)
    return quantity.simplified.magnitude.item()


def convert_unit(x : Array, src_unit : str, dst_unit : str, copy : bool = True) -> Array:
    """Convert the unit of the input array to the destination unit.

    Parameters
    ----------
    x : Array
        Input array.
    src_unit : str
        Source unit.
    dst_unit : str
        Destination unit.
    copy : bool, optional
        Whether to copy the input array. Default is True.
    """
    if isinstance(x, DataArray):
        unit = x.attrs.get('unit', None)
        if unit:
            if not units_equal(unit, src_unit):
                raise ValueError(f"DataArray has unit '{display_unit(unit)}', expected '{display_unit(src_unit)}'")
        else:
            warnings.warn(f"DataArray has no unit, assuming '{display_unit(src_unit)}'")
    scale_factor = conversion_factor(src_unit, dst_unit)
    if copy:
        x = x * scale_factor
    else:
        x *= scale_factor
    if isinstance(x, DataArray):
        x.attrs['unit'] = dst_unit
    return x

