import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap

"""
Functions for plotting analysis results
"""

def lighten(val, clr, light_scale=0.7, dark_scale=0.6):
    """Change color lightness by value between [0, 1]"""
    clr1 = 1 - (1 - clr) * (1 - light_scale)
    clr2 = clr * (1 - dark_scale)
    val = np.asarray(val)
    if val.size > 1:
        val = val[:, None]
        clr1, clr2 = clr1[None, :], clr2[None, :]
    return (1 - val) * clr1 + val * clr2

def get_lighten_cmap(clr, N=16, light_scale=0.2, dark_scale=0.8, revert=False):
    """Get colormap with varying lightness of a given color"""
    value = np.linspace(1, 0, N) if revert else np.linspace(0, 1, N)
    return LinearSegmentedColormap.from_list('lighten', lighten(value, clr, light_scale, dark_scale))

def plot_multicolor_line(*args, c=None, ax=None, cmap='jet', linewidth=2, linestyle='-', alpha=1):
    """Plot line with varying color on its segments. Input coordinates: x, y, (z, optional)
    args: arrays of coordinates of points
    c: values for color map, corresponding to points or segments
    """
    points = np.column_stack(args)[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if c is None:
        c = np.arange(points.shape[0])
    cmid = (c[:-1] + c[1:]) / 2 if len(c) == points.shape[0] else c
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(c.min(), c.max())
    linecollection = Line3DCollection if segments.shape[-1] > 2 else LineCollection
    lc = linecollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(cmid)
    lc.set_linewidth(linewidth)
    lc.set_linestyle(linestyle)
    lc.set_alpha(alpha)
    if ax is None:
        _, ax = plt.subplots()
    line = ax.add_collection(lc)
    return line

def correlation_plot(x, y, xlabel=None, ylabel=None, xy=(.7, .9), ax=None, **plot_kwargs):
    """Scatter plot with correlation coefficient"""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    kwargs = dict(linestyle='none', marker='.')
    kwargs.update(plot_kwargs)
    ax.plot(x, y, **kwargs)
    ax.set_xlabel(x.name if xlabel is None else xlabel)
    ax.set_ylabel(y.name if ylabel is None else ylabel)
    corr = corr_in_plot(x, y, xy=xy, ax=ax)
    return corr

def corr_in_plot(x, y, xy=(.7, .9), ax=None, hue=None, **kwargs):
    """Plot the correlation coefficient at xy location of axis, by default top right"""
    corr = sp.stats.pearsonr(x, y)
    corr = corr.statistic if hasattr(corr, 'statistic') else corr[0]
    ax = ax or plt.gca()
    ax.annotate(f'p={corr:.3f}', xy=xy, xycoords=ax.transAxes)
    return corr

def aling_axes_limits(axes, axis='xy', include_diag=False):
    """Ensure limits of axes on each pairplot match"""
    lims = np.empty((2, axes.size, 2)) # (xy, axes, limits)
    for i, ax in enumerate(axes.ravel()):
        lims[0, i] = ax.get_xlim()
        lims[1, i] = ax.get_ylim()
    lims = lims.reshape((2, ) + axes.shape + (2, ))
    if not include_diag:
        for i in range(min(axes.shape)):
            lims[:, i, i, 0] = -np.inf
            lims[:, i, i, 1] = np.inf
    align_x, align_y = 'x' in axis, 'y' in axis
    xlims = np.column_stack((lims[0, :, :, 0].max(axis=0), lims[0, :, :, 1].min(axis=0))) if align_x else None
    ylims = np.column_stack((lims[1, :, :, 0].max(axis=1), lims[1, :, :, 1].min(axis=1))) if align_y else None
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            if align_x:
                ax.set_xlim(xlims[j])
            if align_y and (include_diag or i != j):
                ax.set_ylim(ylims[i])

def unit_traits(pca_df, plv, figsize=(6, 4)):
    """Plot principal components, phase locking value and mean firing rate of each unit"""
    n_units = len(pca_df)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.bar(range(n_units), pca_df['projection_on_top_PCs'])
    ax.set_xlim(-1, n_units)
    ax.set_xlabel('units')
    ax.set_ylabel('projection_on_top_PCs, phase_locking_value')
    ax.plot(range(n_units), plv.PLV, color='g', linestyle='none',
            marker='o', markersize=8, markerfacecolor='none', label='PLV')
    ax.plot(range(n_units), plv.PLV_unbiased, color='g', linestyle='none', marker='_', markersize=8, label='unbiased PLV')
    ax.legend(loc='upper center', framealpha=0.2)
    ax2 = ax.twinx()
    ax2.tick_params(axis ='y', labelcolor='r')
    ax2.plot(range(n_units), pca_df['mean_firing_rate'], color='r', linestyle='none',
             marker='o', markersize=8, markerfacecolor='none', label='overall')
    ax2.plot(range(n_units), plv.mean_firing_rate, color='r', linestyle='none',
             marker='_', markersize=8, markerfacecolor='none', label='selected conditions')
    ax2.set_ylabel('mean_firing_rate', color='r')
    ax2.tick_params(axis ='y', labelcolor='r')
    ax2.legend(loc='upper right', framealpha=0.2)
    return fig, ax

def heatmap_in_grid(C, x_bins, y_bins, xticks_fmt=None, yticks_fmt=None, even_grid=True, ax=None, **pcm_kwargs):
    """Plot heatmap in grids
    C: the mesh data used in `matplotlib.axes.Axes.pcolormesh`
    x_bins, y_bins: bin edges of the grids. X, Y used in `matplotlib.axes.Axes.pcolormesh`
    xticks_fmt, yticks_fmt: format string for converting ticks to labels
    even_grid: whether display evenly spaced grids regardless of values of bin edges
        if False, edges of grids are displayed at bin edges locations in the original scale
    ax: axes to display the plot
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    args = [] if even_grid else (x_bins, y_bins)
    pcm = ax.pcolormesh(*args, C, **pcm_kwargs)
    if even_grid:
        ax.set_xticks(range(len(x_bins)))
        ax.set_yticks(range(len(y_bins)))
    else:
        ax.set_xticks(x_bins)
        ax.set_yticks(y_bins)
    xticklabels = x_bins if xticks_fmt is None else map(xticks_fmt.format, x_bins)
    yticklabels = y_bins if yticks_fmt is None else map(yticks_fmt.format, y_bins)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    return pcm

def plot_optotag_units(optotag_df, evoked_ratio_threshold=None, ttest_alpha=None,
                       spike_width_threshold=None, marker_size=3, ax=None):
    """Plot scatter plot of optotag units with evoked ratio and spike width thresholds"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    if ttest_alpha is None:
        label = ''
        ttest_alpha = 0.5
    else:
        label = 't-test alpha'
    alpha_size = -marker_size * np.log2(ttest_alpha)
    if 'p_value' in optotag_df.columns:
        marker_sizes = marker_size * np.clip(-np.log2(optotag_df['p_value']), 1, 30)
    else:
        marker_sizes = alpha_size
    positive = optotag_df['positive']
    colors = np.array(['b', 'r'])[positive.astype(int)]

    ax.scatter(optotag_df['waveform_duration'], optotag_df['evoked_ratio'],
               s=marker_sizes, marker='o', edgecolors=colors, facecolors='none')
    if evoked_ratio_threshold is not None:
        ax.axhline(evoked_ratio_threshold, linestyle=':', color='r', label='evoked ratio threshold')
    if spike_width_threshold is not None:
        ax.axvline(spike_width_threshold, linestyle=':', color='orange', label='spike width threshold')
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ax.scatter(xl[0] - 1, yl[0] - 1, s=alpha_size,
               marker='o', edgecolors='r', facecolors='none', label=label)
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    ax.set_xlabel('Waveform duration (ms)')
    ax.set_ylabel('Evoked ratio')
    ax.legend(loc='upper right', framealpha=0.2)
    return ax
