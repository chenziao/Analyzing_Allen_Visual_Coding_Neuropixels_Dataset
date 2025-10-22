from ..paths.paths import OUTPUT_CONFIG_FILE
from ..utils.config_accessor import ConfigAccessor
from ..utils.quantity_units import DisplayUnit


OUTPUT_CONFIG = ConfigAccessor(OUTPUT_CONFIG_FILE)


"""Display unit for plots

Usage (with 'display_format' set to 'unicode'):
>>> from toolkit.plots.format import UNIT
>>> UNIT.um
'μm'
>>> UNIT['V**2']
'V²'
"""

UNIT = DisplayUnit(OUTPUT_CONFIG['display_unit'])


