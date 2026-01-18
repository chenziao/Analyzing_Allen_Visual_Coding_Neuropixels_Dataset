from pathlib import Path
import matplotlib.pyplot as plt

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


"""Save figures"""

SAVE_FIGURE = OUTPUT_CONFIG['save_figure']
FIGURE_FORMAT = ['.' + ext.lstrip('.') for ext in OUTPUT_CONFIG['figure_format']]


def format_text(s : str, format : str = 'capitalize') -> str:
    """Format text for display. Replay '_' with space and apply format method to the text.

    Parameters
    ----------
    s : str
        Text to format.
    format : str
        Format method to apply to the text.
        Supported methods: 'capitalize', 'title', 'upper', 'lower'.
    """
    return getattr(str, format)(str(s).replace('_', ' '))


def save_figure(
    fig_dir : Path | str,
    figs : plt.Figure | dict[str, plt.Figure],
    name : str = '',
    savefig_kwargs : dict = {},
):
    """Save figures.
    
    Parameters
    ----------
    fig_dir : Path | str
        Directory to save the figures.
    figs : plt.Figure | dict[str, plt.Figure]
        Figure or dictionary of {name: figure} to save.
    name : str
        File name to save the figure when `fig` is a single figure.
    """
    fig_dir = Path(fig_dir)
    savefig_kwargs = OUTPUT_CONFIG['savefig_kwargs'] | savefig_kwargs

    if not isinstance(figs, dict):
        figs = {name: figs}

    for name, fig in figs.items():
        if not name:
            continue
        name.replace('/', '_')
        fig_path = fig_dir / name
        for ext in FIGURE_FORMAT:
            fig.savefig(fig_path.with_suffix(ext), **savefig_kwargs)

