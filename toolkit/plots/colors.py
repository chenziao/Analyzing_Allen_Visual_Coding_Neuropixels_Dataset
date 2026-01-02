import matplotlib.pyplot as plt
import numpy as np


BAND_COLORS = {
    'beta': 'b',
    'gamma': 'r',
    'LGN gamma': 'y',
    None: 'gray',
}


# VISp layer colors for convenience
VISP_LAYER_ACRONYMS = ['1', '2/3', '4', '5', '6a', '6b']
N_VISP_LAYERS = len(VISP_LAYER_ACRONYMS)
VISP_LAYER_COLORS = plt.get_cmap('plasma', N_VISP_LAYERS)(np.linspace(0, 1, N_VISP_LAYERS))
VISP_LAYER_COLORS_DICT = dict(zip(VISP_LAYER_ACRONYMS, VISP_LAYER_COLORS))
