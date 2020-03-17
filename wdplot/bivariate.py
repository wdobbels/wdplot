import numpy as np
import pandas as pd
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from .prep import estimate_density, sliding_window

nopython = True

def hexbin(x, y, c, ax=None, minalpha=0.05, maxalpha=1., scaleval=10.,
                log_dens=True, f_aggregate=np.mean, **hexbin_kwargs):
    """
    Make a 2D hex plot, with a given color. Transparency is density.

    Note: this changes the matplotlib backend to 'agg', since it does not
    seem to work for other backends.
    
    Parameters
    ----------

    minalpha : float, default 0.05
        Minimum transparency value (for lowest count)
    maxalpha : default 1.
        Maximum transparency value. Can be put > 1, which results in saturation.
    """
    
    if mpl.get_backend() != 'agg':
        print('Switching matplotlib backend to agg')
        mpl.use('agg')
    # Vary only hue as default (luminosity for transparency)
    cmap_hue = mpl.colors.ListedColormap(sns.husl_palette(256, l=0.5)[:179][::-1])
    hexbin_kwargs.setdefault('cmap', cmap_hue)
    hexbin_kwargs.setdefault('mincnt', 1)
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure
    # Get counts per bin
    sdens = ax.hexbin(x, y, C=c, reduce_C_function=len, alpha=0.001,
                      **hexbin_kwargs)
    counts = sdens.get_array()  # size: (n_bins, )
    sdens.remove()
    # Hexbin with the right color (but wrong alpha)
    s = ax.hexbin(x, y, C=c, reduce_C_function=f_aggregate, alpha=0.99,
                  **hexbin_kwargs)
    plt.draw()
    # size: (n_bins, 4)  (rgba for each cell)
    # note: this size only works for the 'agg' backend. With other
    #       (interactive) backends, this has shape (1, 4), and this can not
    #       be fixed by calling draw and draw_idle.
    cols = s.get_facecolors()
    # counts = counts + (np.max(counts)*tresfrac)  # lower bound for alpha
    if log_dens:
        if scaleval > 0:
            counts = counts / np.max(counts)
            counts = log_scale(counts, scaleval)
        else:
            counts = simple_log(counts)
    else:
        counts = counts / np.max(counts)
    # counts is now between 0 and 1: rescale the alpha
    counts = counts * (maxalpha - minalpha) + minalpha
    counts = np.clip(counts, 0, 1)
    cols[:, 3] = counts  # change alpha
#         cols[:, :3] = counts[:, None]*cols[:, :3]  # blend white
    # Edges are not transparent enough: reduce edge strength
    edgecols = cols.copy()
    edgecols[:, 3] = edgecols[:, 3] / 5.
    # SET EVERYTHING (can probably delete some of these lines)
    s.set_facecolor(cols)
    s.set_edgecolor(edgecols)
    s._facecolors = cols
    s._edgecolors = edgecols
    f.canvas.draw()
    f.canvas.draw_idle()
    return s

def hist2d(x, y, c, bins=30, cmap=plt.cm.viridis, ax=None):
    '''
    Create a 2d histogram plot, but instead of coloring by density we
    determine the transparancy by point density (binned). The color is an
    extra variable, for which the binned value takes an average.

    Examples
    --------

    >>> n = 10000
    >>> x = np.log10(np.abs(np.random.randn(n) + 2))
    >>> x[x < -1.5] = 0
    >>> y = np.random.randn(n)
    >>> c = np.power(10, x) + y
    >>> hist2d(x, y, c, bins=40)
    
    '''

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(c, pd.Series):
        c = c.values
    b_sel = (np.isfinite(x) & np.isfinite(y) & np.isfinite(c))
    x, y, c = x[b_sel], y[b_sel], c[b_sel]
    # useful for making the edges, and you get the density as a bonus
    dens, xedges, yedges = np.histogram2d(x, y, bins=bins)
    # Create something similar to dens (2D binned array), but using average color
    arr_c = create_color_array(x, y, c, xedges, yedges, dens.astype(np.int))
    # convert to rgba array of shape (Nx, Ny, 4)
    color = mpl.colors.Normalize()(arr_c)
    color = cmap(color)
    color[..., -1] = dens.T / np.max(dens)
    # create plot
    if ax is None:
        f, ax = plt.subplots()
    ax.set_facecolor((1, 1, 1))
    ax.imshow(color, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
              origin='lower', aspect='auto')
    return ax

def scatter_trendline(x, y, ax=None, kw_sliding_window=None,
                      kw_scatter=None, kw_line=None):
    """
    Make scatterplot with a sliding window trendline (see 
    `wdplot.prep.sliding_window`).
    """

    if ax is None:
        ax = plt.gca()
    if kw_sliding_window is None:
        kw_sliding_window = {}
    if kw_scatter is None:
        kw_scatter = {}
    if kw_line is None:
        kw_line = {}

    x_trend, y_trend = sliding_window(x, y, **kw_sliding_window)
    # Plotting colour as density
    if kw_scatter.get('c', 'density') == 'density':
        x, y, c = estimate_density(x, y)
        kw_scatter['c'] = c
    ax.scatter(x, y, **kw_scatter)
    ax.plot(x_trend, y_trend, **kw_line)
    return x_trend, y_trend

def scatter_trendline_points(x, y, binsize=101, func=np.mean, ax=None,
                      kw_scatter=None, kw_line=None):
    """
    Scatter with sliding window trendline, but unlike `scatter_trendline`
    the window (both center and size) is determined by the point indices.
    """

    if (binsize % 2) == 0:
        print('scatter_trendline: uneven binsize not supported!')
        return
    if ax is None:
        ax = plt.gca()
    if kw_scatter is None:
        kw_scatter = {}
    if kw_line is None:
        kw_line = {}

    bin_halfw = binsize // 2
    idx = bin_halfw
    idx_sort = np.argsort(x)
    x_sort = x.copy()[idx_sort]
    y_sort = y.copy()[idx_sort]
    idx_max = x.shape[0] - 1 - bin_halfw
    n_trend = x.shape[0] - (binsize - 1)
    x_trend = np.zeros(n_trend)
    y_trend = np.zeros(n_trend)
    idx_trend = 0
    # Loop over all points that can cover the bin width
    while idx <= idx_max:
        x_trend[idx_trend] = x_sort[idx]
        y_trend[idx_trend] = func(y_sort[idx_trend-bin_halfw:idx_trend+bin_halfw])
        idx += 1
        idx_trend += 1
    # Plotting colour as density
    if kw_scatter.get('c', 'density') == 'density':
        x, y, c = estimate_density(x, y)
        kw_scatter['c'] = c
    ax.scatter(x, y, **kw_scatter)
    ax.plot(x_trend, y_trend, **kw_line)
    return x_trend, y_trend


# ------ HELPER FUNCTIONS -------
@jit(nopython=nopython)
def avg_color(ix, iy, xedges, yedges, x, y, c, nbins):
    xl, xr = xedges[ix], xedges[ix+1]
    yl, yr = yedges[iy], yedges[iy+1]
    li_c = np.zeros(nbins)
    j = 0
    for i, cpoint in enumerate(c):
        if (x[i] >= xl) and (x[i] <= xr) and (y[i] >= yl) and (y[i] <= yr):
            li_c[j] = cpoint
            j += 1
    if j > 0:
        # set_trace()
        return np.mean(li_c)
    return 0

@jit(nopython=nopython)
def create_color_array(x, y, c, xedges, yedges, dens):
    arr_c = np.zeros(dens.shape)
    for iy in range(arr_c.shape[0]):
        for ix in range(arr_c.shape[1]):
            arr_c[iy, ix] = avg_color(ix, iy, xedges, yedges, x, y, c, dens[ix, iy])
    return arr_c

def log_scale(x, a=100):
    return np.log10(1+(a*x)) / np.log10(1+a)

def simple_log(x):
    return np.log10(x) / np.log10(x.max())