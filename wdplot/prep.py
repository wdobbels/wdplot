'''
Functions that are usually run in preparation of a plot, to aid in binning,
colouring, ...
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def sliding_window(x, y, binwidth=None, minpoints=80, func=np.mean):
    """
    Runs a sliding window around the datapoints. The window size is defined
    in physical units, but contains a minimum number of datapoints.
    """

    bin_probe_factor = 1.1
    idx_s = np.argsort(x)
    x, y = x[idx_s], y[idx_s]
    xmax = x[-1]
    
    if binwidth is None:
        binwidth = (xmax - x[0]) / 60
    binhw = binwidth / 2
    xc = x[0]
    x_bin = []
    y_bin = []
    j = 0
    prev_center = xc - 1
    while xc - binhw < xmax:
        b_bin = 0
        ext_binhw = binhw / bin_probe_factor
        # Ensure enough points in bin
        i = 0
        while np.sum(b_bin) < minpoints:
            ext_binhw = ext_binhw * bin_probe_factor
            b_bin = (x < xc + ext_binhw) & (x >= xc - ext_binhw)
            i += 1
            if i > 100:
                import pdb
                pdb.set_trace()
        x_center = np.mean(x[b_bin])
        # When we start going back (after increasing bin), usually shitty,
        # but only break if we're close to the end (> 80%)
        if (x_center < prev_center) and (x_center > (0.8*xmax + 0.2*x[0])):
            break
        prev_center = x_center
        x_bin.append(x_center)
        y_bin.append(func(y[b_bin]))
        eff_binw = x[b_bin][-1] - x[b_bin][0]
        if eff_binw < binhw:
            eff_binw = binhw
        xc = xc + eff_binw / 4  # Step size: quarter of effective bin width
        j += 1
        if j > 1000:
            pdb.set_trace()
    # Can be in wrong order due to bin increase (not a lot of points)
    x_bin, y_bin = np.array(x_bin), np.array(y_bin)
    idx_s = np.argsort(x_bin)
    return x_bin[idx_s], y_bin[idx_s]

def estimate_density(x, y, method='scott', sort_points=True, subsample=None,
                     logspace=False, **kwargs):
    """
    Useful for scatterplots. This makes it possible to plot high density 
    regions with another color instead of just having overlapping dots.

    Parameters
    ----------
    x, y : array-like
        These are combined into a 2D vector to estimate the density.
    method : str
        Guassian kde estimation technique. Supports all values of
        `scipy.stats.gaussian_kde`, as well as 'sklearn'. This method in
        turn calculates an optimal kernel size. 'sklearn' is slower (due to the
        grid search), but recommended by [jake vanderplas](https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/).
    sort_points : bool
        Reorders x and y so high density points come on top. 
    subsample : int or None
        Take a subsample of the points, in order to avoid
        long calculations. Above 10 000 points, the calculation takes some 
        time. 
    logspace : bool, default False
        If True, x and y are transformed to log space before calculating the
        density, and are then transformed back.

    Returns
    -------
    x, y, density : array-like
        These match index by index. x and y are returned because they can be 
        reordered or resampled.

    Examples
    --------
    
    >>> x, y = np.random.randn(500), np.random.randn(500)
    >>> xc, yc, c = estimate_density(x, y)
    >>> plt.scatter(xc, yc, c=c, alpha=0.3)

    """

    def kde_sklearn(xy):
        from sklearn.neighbors import KernelDensity
        from sklearn.model_selection import GridSearchCV, LeaveOneOut

        kwargs.setdefault('cv', None)
        kwargs.setdefault('refit', False)
        # Sample bandwidth as log-uniform
        bandwidths = np.logspace(-1, 1, kwargs.pop('n_bandwidths', 30))
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            **kwargs)
        grid.fit(xy)
        bw = grid.best_params_['bandwidth']
        kde = KernelDensity(bandwidth=bw).fit(xy)
        z = kde.score_samples(xy)
        return z

    # Convert pandas series to numpy array
    if isinstance(x, pd.core.series.Series):
        x = x.values
    if isinstance(y, pd.core.series.Series):
        y = y.values
    assert x.shape == y.shape, "x and y must have equal lengths"
    if logspace:
        x, y = np.log(x), np.log(y)
    if subsample is not None:
        idx = np.random.choice(x.shape[0], size=subsample, replace=False)
        x, y = x[idx], y[idx]
    xy = np.vstack([x, y])
    if method == 'sklearn':
        z = kde_sklearn(xy.T)
    else:
        kde = gaussian_kde(xy, bw_method=method, **kwargs)
        z = kde(xy)

    if logspace:
        x, y = np.exp(x), np.exp(y)
    if not sort_points:
        return x, y, z
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    return x[idx], y[idx], z[idx]

def continuous_to_discrete_cmap(ncolors, cmap_name='gnuplot', vmin=0., vmax=1.):
    """Get a list of ncolors colors, sampled uniformly from a continuous cmap."""

    cmap = plt.cm.get_cmap(cmap_name)
    return [cmap(np.interp(i, [0, ncolors-1], [vmin, vmax])) for i in range(ncolors)]