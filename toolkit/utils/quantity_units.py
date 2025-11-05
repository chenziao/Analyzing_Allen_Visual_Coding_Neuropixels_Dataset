"""
Unit conversion utilities.

Usage:
>>> from toolkit.utils.quantity_units import convert_unit
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
    """Convert a unit string to a unit quantity object."""
    return pq.unit_registry[unit]


def unit_to_str(quantity : UnitQuantity, use_unicode : bool = False) -> str:
    """Convert a unit quantity object to a string."""
    with unicode_units(use_unicode):
        dim = str(quantity.dimensionality)
        return '' if dim == 'dimensionless' else dim


def as_quantity(unit : str | UnitQuantity) -> UnitQuantity:
    """Convert a unit string or unit object to a unit quantity object."""
    return str_to_unit(unit) if isinstance(unit, str) else unit


def as_string(unit : str | UnitQuantity) -> str:
    """Convert a unit to string format"""
    try:
        return unit_to_str(as_quantity(unit), use_unicode=False)
    except:
        return str(unit)


def as_unicode(unit : str | UnitQuantity) -> str:
    """Convert the unit to unicode format"""
    try:
        return unit_to_str(as_quantity(unit), use_unicode=True)
    except:
        return str(unit)


def as_latex(unit : str | UnitQuantity) -> str:
    """Convert the unit to latex format"""
    try:
        return as_quantity(unit).dimensionality.latex
    except:
        return str(unit)


def units_equal(unit1 : str | UnitQuantity, unit2 : str | UnitQuantity) -> bool:
    """Check if two units are equal."""
    try:
        return as_quantity(unit1) == as_quantity(unit2)
    except Exception as e:
        if isinstance(unit1, str) and isinstance(unit2, str):
            return unit1 == unit2
        else:
            raise e


def conversion_factor(src_unit : str | UnitQuantity, dst_unit : str | UnitQuantity) -> float:
    """Get the conversion factor between two units."""
    quantity = as_quantity(src_unit) / as_quantity(dst_unit)
    return quantity.simplified.magnitude.item()


def convert_unit(
    x : Array,
    src_unit : str | UnitQuantity,
    dst_unit : str | UnitQuantity, 
    copy : bool = False
) -> Array:
    """Convert the unit of the input array to the destination unit.

    Parameters
    ----------
    x : Array
        Input array.
    src_unit : str | UnitQuantity
        Source unit.
    dst_unit : str | UnitQuantity
        Destination unit.
    copy : bool, optional
        Whether to copy the input array. If False, the input array will be modified in place. Default is False.
    """
    if isinstance(x, DataArray):
        unit = x.attrs.get('unit', None)
        if unit:
            if not units_equal(unit, src_unit):
                raise ValueError(f"DataArray has unit '{as_unicode(unit)}', expected '{as_unicode(src_unit)}'")
        else:
            warnings.warn(f"DataArray has no attribute 'unit', assuming unit '{as_unicode(src_unit)}'")
    scale_factor = conversion_factor(src_unit, dst_unit)
    if copy:
        x = x * scale_factor
    else:
        x *= scale_factor
    if isinstance(x, DataArray):
        x.attrs['unit'] = dst_unit
    return x


class DisplayUnit:
    """Display unit for plots."""
    def __init__(self, format : str = 'unicode'):
        match format:
            case 'unicode':
                self.display = as_unicode
            case 'latex':
                self.display = as_latex
            case 'string':
                self.display = as_string
            case _:
                raise ValueError(f"Invalid format: {format}")

    def __getattr__(self, unit : str) -> str:
        return self.display(unit)

    def __getitem__(self, unit : str | UnitQuantity) -> str:
        return self.display(unit)
