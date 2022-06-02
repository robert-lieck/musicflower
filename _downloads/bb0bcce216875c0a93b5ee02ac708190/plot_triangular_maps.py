"""
Triangular Maps
===============
"""

# %%
# The array returned by :func:`~musicflower.loader.load_file` contains the triangular map of pitch-class distributions.
# To use NumPy's efficient linear storage, it is flattened by starting at the top and then going row by row, thus, it
# contains a total number of :math:`n(n+1)/2` pitch-class distributions.
from musicflower.loader import load_file
resolution = 10
scape = load_file(data='Prelude_No._1_BWV_846_in_C_Major.mxl', n=resolution)
print(scape.shape)
print(resolution * (resolution + 1) / 2)

# %%
# To handle this kind of flattened triangular maps, the :class:`~triangularmap.tmap.TMap` class from the
# `TriangularMap <https://github.com/robert-lieck/triangularmap>`_ package can be used. It takes
# a linear storage and allows for efficiently accessing its content as if it was a triangular map (see the
# `TriangularMap documentation <https://robert-lieck.github.io/triangularmap/>`_ for more details).
# For multidimensional arrays, the first dimension is assumed to represent the triangular map.

from triangularmap import TMap
import numpy as np
tmap = TMap(np.arange(6))
print(tmap.pretty())

# %%
# The functionalities include slicing at a specific level/depth (this returns *views* of the underlying storage)
print(tmap.dslice(0))  # depth starts a 0 from the top
print(tmap.lslice(1))  # levels start at 1 from the bottom

# %%
# and slicing "skewed columns" for a specific start/end index (this returns *copies* of the underlying storage, note the
# different syntax using square brackets), which always run left-to-right (i.e. bottom-up for start-slices, top-down for
# end-slices)
#
print(tmap.sslice[0])  # start indices run from 0 to n - 1
print(tmap.eslice[3])  # end indices run from 1 to n
