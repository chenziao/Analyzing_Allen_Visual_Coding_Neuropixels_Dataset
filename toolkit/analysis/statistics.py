import numpy as np

from typing import Callable
from numpy.typing import NDArray, ArrayLike


class WeightedPCA:
    """Weighted PCA class (Allow fitting data with weights)"""

    def __init__(self, n_components):
        """Initialize the WeightedPCA object

        Parameters
        ----------
        n_components : int
            Number of principal components to keep.

        Attributes
        ----------
        n_components : int
            Number of principal components to keep.
        components_ : NDArray[float]
            Principal components matrix (n_components, n_features).
        mean_ : NDArray[float]
            Weighted mean of the data (n_features,).
        explained_variance_ratio_ : NDArray[float]
            Explained variance ratio (n_components,).
        """
        self.n_components : int = n_components
        self.components_ : NDArray[float] | None = None
        self.mean_ : NDArray[float] | None = None
        self.explained_variance_ratio_ : NDArray[float] | None = None

    def fit(self, X : NDArray[float], weights : NDArray[float] | None = None) -> 'WeightedPCA':
        """Fit the WeightedPCA model

        Parameters
        ----------
        X : NDArray[float]
            Data (n_samples, n_features) matrix to fit the model.
        weights : NDArray[float], optional
            Weights (n_samples,) for the data.
            If not provided, weights are equal for all samples (same as sklearn.decomposition.PCA).

        Returns
        -------
        self : WeightedPCA
            Fitted WeightedPCA model.
        """
        X = np.asarray(X)
        if weights is None:
            weights = np.ones(X.shape[0])
        else:
            weights = np.asarray(weights)
            if X.shape[0] != weights.size:
                raise ValueError("Size of `weigths` does not match number of samples in `X`.")
        if X.shape[1] < self.n_components:
            raise ValueError("Number of features in `X` is less than `n_components`.")

        # Weighted mean and centering
        self.mean_ = np.average(X, axis=0, weights=weights)
        X_centered = X - self.mean_

        # Apply square root weights for SVD
        # This makes X.T @ W @ X = (X * sqrt(W)).T @ (X * sqrt(W))
        X_weighted = X_centered * np.sqrt(weights)[:, np.newaxis]

        # SVD on weighted, centered data
        _, S, Vt = np.linalg.svd(X_weighted, full_matrices=False)
        self.components_ = Vt[:self.n_components]

        # Explained variance
        vars = S ** 2
        self.explained_variance_ratio_ = vars / np.sum(vars)
        return self

    def transform(self, X : NDArray[float]) -> NDArray[float]:
        """Transform data using the fitted WeightedPCA model

        Parameters
        ----------
        X : NDArray[float]
            Data (n_samples, n_features) matrix to transform.

        Returns
        -------
        X_transformed : NDArray[float]
            Transformed data (n_samples, n_components) matrix.
        """
        # Subtract the fitted mean and project onto components
        if self.components_ is None:
            raise ValueError("This PCA instance is not fitted yet.")
        return (np.asarray(X) - self.mean_) @ self.components_.T

    def fit_transform(self, X : NDArray[float], weights : NDArray[float] | None = None) -> NDArray[float]:
        """Fit the WeightedPCA model and transform the same data"""
        self.fit(X, weights)
        return self.transform(X)


def quantize(x : ArrayLike, n_bins : int) -> tuple[NDArray[int], NDArray[float]]:
    """Quantize 1-d data by its equally spaced quantiles

    Parameters
    ----------
    x : ArrayLike
        Data array. Array is flattened to get the quantiles if it is not 1-d array.
    n_bins : int
        Number of quantile bins of equal space.

    Returns
    -------
    bid : NDArray[int]
        Array of bin index valued in [0, ..., n_bins - 1] of data points (same shape as `x`).
    bins : NDArray[float]
        Values of (n_bins + 1,) quantile bin edges.
    """
    x = np.asarray(x)
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bid = np.digitize(x, bins[1:-1])
    return bid, bins


def quantize_nd(
    X : ArrayLike,
    n_bins : int | list[int] = 8
) -> tuple[NDArray[NDArray[int]], tuple[NDArray[float]], NDArray[int]]:
    """Quantize n-d data into grids by equally spaced quantiles of each feature

    Parameters
    ----------
    X : ArrayLike
        Data (n_features, n_samples) matrix to be quantized.
        If array dimension is greater than 2, all dimensions except the first one are flattened.
    n_bins : int | list[int]
        Number of quantile bins [m_0, m_1, ..., m_{p-1}] of equal space for each feature.
        If an integer, the same number of bins is used for all features.

    Returns
    -------
    idx_in_grid : NDArray[NDArray[int]]
        Indices of data points that belong to each quantile grid of features.
        An (m_0, m_1, ..., m_{p-1}) array, each element of which is an indices array.
    bins : tuple[NDArray[float]]
        Tuple of (m_i + 1,) arrays of quantile bin edges for each feature.
    hist_counts : NDArray[int]
        Histogram counts in each quantile grid (m_0, m_1, ..., m_{p-1}).
    """
    X = np.asarray(X).reshape(len(X), -1)  # (p, n) matrix of p-features with sample dimension flattened
    if isinstance(n_bins, int):  # p features have the same m=bins
        n_bins = [n_bins] * X.shape[0]
    else:
        if len(n_bins) != X.shape[0]:  # p features have different m_i bins (i=0, ..., p-1)
            raise ValueError("Size of `n_bins` should match number of features `X`")

    # Quantize each feature into grids
    bids, bins = zip(*map(quantize, X, n_bins))  # p-lists of bin indices (n,) and bin edges (m_i + 1,)
    bidx = [np.arange(n) for n in n_bins]  # p-lists of bin indices arrays (m_i,)
    gids = [bid == idx[:, None] for bid, idx in zip(bids, bidx)]  # p-lists of (m_i, n) grid boolean indices matrices
    grid_idx = np.meshgrid(*bidx, indexing='ij')  # p-lists of grid arrays (m_0, m_1, ..., m_{p-1}) of indices along each feature

    # Count and mark data points in each grid
    idx_in_grid = np.full(n_bins, None, dtype=object)  # (m_0, m_1, ..., m_{p-1}) array of data indices in each grid
    for gidx in np.nditer(grid_idx):  # idx is a p-tuple of indices along each feature (i_0, i_1, ..., i_{p-1})
        idx = np.all([gid[i] for gid, i in zip(gids, gidx)], axis=0)  # (n,) array of boolean indices of data points in the grid
        idx_in_grid[gidx] = np.nonzero(idx)[0]  # indices array of data points in the grid
    hist_counts = np.vectorize(np.size)(idx_in_grid)  # histogram counts in each grid
    return idx_in_grid, bins, hist_counts


def statistic_in_grid(
    X : ArrayLike,
    idx_in_grid : NDArray[NDArray[int]],
    hist_counts : NDArray[int] | None = None,
    stat_method : Callable | str = np.mean,
    stat_fill : float = np.nan
) -> tuple[NDArray[float], NDArray[int]]:
    """Get statistics of data features for samples divided into each of n-d grids

    Parameters
    ----------
    X : ArrayLike
        Data (n_features, n_samples) matrix to get statistics for.
        If array dimension is greater than 2, all dimensions except the first one are flattened.
    idx_in_grid : NDArray[NDArray[int]]
        Indices of data points that belong to each quantile grid of features (as returned by `quantize_nd`).
        An (m_0, m_1, ..., m_{p-1}) array, each element of which is an indices array.
    hist_counts : NDArray[int], optional
        Histogram counts in each grid. If not provided, it is caculated from `idx_in_grid`.
    stat_method : Callable | str
        Function to calculate statistics for each feature. If a string, it is a numpy function name like 'mean'.
    stat_fill : float, optional
        Value to fill when no data exists in a grid. Default: nan.

    Returns
    -------
    stats : NDArray[float]
        Statistics of each feature for each grid (n_features, m_0, m_1, ..., m_{p-1}).
    hist_counts : NDArray[int]
        Histogram counts in each grid (m_0, m_1, ..., m_{p-1}).
    """
    if hist_counts is None:
        hist_counts = np.vectorize(np.size)(idx_in_grid)
    if isinstance(stat_method, str):
        stat_method = getattr(np, stat_method)
    elif not callable(stat_method):
        raise ValueError("`stat_method` must be a callable function or a string of numpy function name like 'mean'.")
    X = np.asarray(X).reshape(len(X), -1)  # (p, n) matrix of p-features with sample dimension flattened
    stats = np.full(X.shape[:1] + idx_in_grid.shape, stat_fill, dtype=type(stat_fill))
    for x, stat in zip(X, stats):
        for idx in np.ndindex(idx_in_grid.shape):
            if hist_counts[idx]:
                stat[idx] = stat_method(x[idx_in_grid[idx]])
    return stats, hist_counts


def range_by_iqr(X : ArrayLike, n_iqr : float | list[float] = 1.5, axis : int | None = None) -> NDArray[float]:
    """Value range limited by outliers in term of multiple of interquartile
    
    Parameters
    ----------
    X : ArrayLike
        Data array. Array is flattened to get the quantiles if it is not 1-d array.
    n_iqr : float | list[float]
        Multiple of interquartile to limit the lower and upper value range.
        If a single value, the same value is used for both lower and upper value range.
    axis : int | None
        Axis along which to calculate the value range. If None, the value range is calculated for the entire array.

    Returns
    -------
    vlim : NDArray[float]
        Value range limited by outliers in term of multiple of interquartile. Shape (2, ...).
        The first axis of the result corresponds to the lower and upper value range.
        The other axes are the axes that remain after the reduction of `axis`.
    """
    X = np.asarray(X)
    n_iqr = np.broadcast_to(n_iqr, 2)
    q1, q3 = np.nanquantile(X, [0.25, 0.75], axis=axis)
    r1, r3 = np.tensordot(n_iqr, q3 - q1, axes=0)  # like outer product
    vlim = np.fmax(np.nanmin(X, axis=axis), q1 - r1), np.fmin(np.nanmax(X, axis=axis), q3 + r3)
    return np.stack(vlim, axis=0)
