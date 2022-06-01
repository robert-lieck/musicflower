"""
Extract Data
===========================

The relevant data can be extracted as described below, for instance, if a different visualisation backend than Plotly
should be used (see :doc:`plot_basic_example` for basic functionality).
"""

# %%
# Loading Files
# ------------------------
# There are some options for loading files. First, the :func:`~musicflower.loader.load_file` function
# provides a number of parameters. Second, the :func:`~musicflower.loader.load_corpus` function can be used
# to parallelise loading of multiple files (see documentation for more details). Here, we simply load the same file
# twice, parallelise loading, and cache results (subsequent runs of this script will reuse the cached
# result from the first run). The resulting corpus corresponds to a concatenation of the outputs generated by the
# :func:`~musicflower.loader.load_file` function with the number of files as its first dimension. Note that the
# `file_paths` (and scapes in the corpus) may have a different order than the input `file_paths` (see documentation of
# :func:`~musicflower.loader.load_corpus` for details).
import numpy as np

from musicflower.loader import load_corpus
file_path = 'Prelude_No._1_BWV_846_in_C_Major.mxl'
resolution = 50
corpus, file_paths = load_corpus(
    file_paths=[file_path, file_path],
    n=resolution,
    parallel=True,
    use_cache=True
)
print(corpus.shape)

# %%
# Colours and 3D Coordinates
# --------------------------
# These data can be mapped to colours and 3D coordinates as shown in the :doc:`plot_basic_example`, that is

from musicflower.plotting import key_colors
from musicflower.util import get_fourier_component, remap_to_xyz
colors = key_colors(corpus)
x, y, z = remap_to_xyz(*get_fourier_component(pcds=corpus, fourier_component=5))
print(colors.shape)
print(x.shape)

# %%
# Triangular Surface
# ------------------
# The points in 3D space define a triangular surface, which corresponds the surface shown in a key scape plot "folded
# and wrapped" in 3D space. The point indices of the respective triangles can be obtained with the
# :func:`~musicflower.util.surface_scape_indices` function. The resolution can either be explicitly specified or via an
# array dimension.

from musicflower.util import surface_scape_indices
i, j, k = surface_scape_indices(colors, axis=1)  # from array dimension
i_, j_, k_ = surface_scape_indices(resolution)   # from resolution
np.array_equal([i, j, k], [i_, j_, k_])          # result is the same

# %%
# The triangle with a specific index can be obtained as

piece = 0
idx = 0
print("1st vertex (i):", x[piece, i[idx]], y[piece, i[idx]], z[piece, i[idx]])
print("2nd vertex (j):", x[piece, j[idx]], y[piece, j[idx]], z[piece, j[idx]])
print("3rd vertex (k):", x[piece, k[idx]], y[piece, k[idx]], z[piece, k[idx]])

# %%
# or, more compactly, by concatenating coordinates and indices and using NumPy's advanced indexing

xyz = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=-1)
ijk = np.concatenate((i[..., None], j[..., None], k[..., None]), axis=-1)
print(xyz[piece, ijk[idx]])

# %%
# Time Traces
# -----------
# In :doc:`plot_basic_example` the time traces were plotted, but we did not extract the actual data for them.
# They are computed by interpolating along rows of the triangular map using the
# :func:`~musicflower.util.time_traces` function. In the output, the first dimension are time steps, the second runs
# along the traces (from the top of the triangle to the bottom), and the last one are xyz-coordinates or RGB-colours,
# respectively. The third (batch) dimension are the different pieces in this case.

from musicflower.util import time_traces
xyz_traces, colors_traces = time_traces(x=x, y=y, z=z, colors=colors, n_steps=200, axis=1)
print(xyz_traces.shape, colors_traces.shape)



