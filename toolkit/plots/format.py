import json

from ..paths.paths import OUTPUT_CONFIG_FILE
from ..utils.quantity_units import DisplayUnit


with open(OUTPUT_CONFIG_FILE, 'r') as f:
    OUTPUT_CONFIG = json.load(f)


"""Display unit for plots

Usage (with 'display_format' set to 'unicode'):
>>> from toolkit.plots.format import UNIT
>>> UNIT.um
'μm'
>>> UNIT['V**2']
'V²'
"""

UNIT = DisplayUnit(OUTPUT_CONFIG['display_unit'])


