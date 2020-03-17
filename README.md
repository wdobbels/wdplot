WDPlot is a plotting library on top of matplotlib. The library places special
importance on avoiding overplotting in scatterplots. The following functions
can help to solve this problem:

- `wdplot.prep.estimate_density`: does a kernel density estimate, alowing for a scatter plot colored by density.
- `wdplot.bivariate.hexbin`: does a hexbin plot, where the transparency denotes density. Coloring can be a third variable or also density.
- `wdplot.bivariate.hist2d`: same as above, but with a 2D hist (square bins)
- `wdplot.prep.sliding_window`: does a sliding window estimate of a (x, y)-relation
- `wdplot.bivariate.scatter_trendline`: makes a scatter-plot with a sliding window trendline on top

# Install

Using pip:
`python -m pip install wdplot`

# Current state

This package is mostly a collection of plot functions I tend to reuse during my research. There's no tests, and no guarantees for a stable API. Of course, contributions are welcome, but since the code was written for myself, there is much left to be desired regarding clean code and documentation.