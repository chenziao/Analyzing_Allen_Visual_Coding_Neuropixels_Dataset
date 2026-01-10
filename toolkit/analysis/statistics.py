import numpy as np

from typing import Self
from numpy.typing import NDArray


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

    def fit(self, X : NDArray[float], weights : NDArray[float] | None = None) -> Self:
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
        Self
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


def quantize(x, n_bins):
    """Quantize data in an array by its equally spaced quantiles
    x: data array
    n_bins: number of quantile bins of equal space
    Return: array of bin index of data points, value of bin edges
    """
    x = np.asarray(x)
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bid = np.digitize(x, bins[1:-1])
    return bid, bins


def statistic_in_quantile_grid(X, Y, n_bins=8, stat=np.mean, stat_fill=np.nan):
    """Divide data points into grids by n-quantiles of some features and
    calculate statistics of some features in the grids
    X: list of arrays of features according to which data are divided
    Y: list of arrays of features of which to obtain statistics
    n_bins: number of bins for all features in X
        or a list of them corresponding to each feature in X
    stat: function that calculate a statistic of each feature in Y. default: mean
        function should allow operation along specific axis with argument `axis`
    stat_fill: value to fill when no data exists in a grid. default: nan
    Return: statistics of each feature in Y, bin edges of each features in X, nd histogram count
    Note: data in X and Y must not contain nan values
    """
    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(X)
    else:
        if len(X) != len(n_bins):
            raise ValueError("Size of `n_bins` should match number of features `X`")
    bids, bins = zip(*map(quantize, X, n_bins))
    bidx = [np.arange(n) for n in n_bins]
    gids = [bid == idx[:, None] for bid, idx in zip(bids, bidx)]
    grid_ids = np.meshgrid(*bidx, indexing='ij')
    idx_in_grid = np.full(n_bins, None, dtype=object)
    hist_count = np.zeros(n_bins, dtype=int)
    for ids in np.nditer(grid_ids):
        idx = np.all([gid[i] for gid, i in zip(gids, ids)], axis=0)
        idx_in_grid[ids] = np.nonzero(idx)[0]
        hist_count[ids] = idx_in_grid[ids].size
    y_stats = []
    for y in Y:
        y = np.asarray(y)
        y_stat = np.full(n_bins, stat_fill, dtype=y.dtype)
        for ids in np.nditer(grid_ids):
            if hist_count[ids]:
                y_stat[ids] = stat(y[idx_in_grid[ids]])
        y_stats.append(y_stat)
    y_stats = np.stack(y_stats, axis=0)
    return y_stats, bins, hist_count
