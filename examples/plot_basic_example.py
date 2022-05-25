"""
Basic Example
===========================

This is a brief walk through some basic MusicFlower functionality.
"""

# %%
# Loading a File
# ------------------------
#
# We are using a MusicXML file as it is small enough to ship with the documentation.
# It can be loaded using the :meth:`~musicflower.loader.load_file` function

from musicflower.loader import load_file

# path to file
file_path = 'Prelude_No._1_BWV_846_in_C_Major.mxl'
# split piece into this many equal-sized time intervals
resolution = 50
# get pitch scape at specified resolution
scape = load_file(file_path=file_path, n=resolution)

# %%
# The result can be visualised in a key scape plot.
from musicflower.plotting import plot_key_scape
plot_key_scape(scape)

# %%
# More functionality, such as a legend for the used colours, is available via the PitchScapes library
import pitchscapes.plotting as pt
_ = pt.key_legend()

# %%
# The array returned by :meth:`~musicflower.loader.load_file` contains the triangular map of pitch-class distributions
# (PCDs) that is visualised above. To use NumPy's efficient linear storage, it is flattened by starting at the top and
# then going row by row, thus, it contains a total number of n(n+1)/2 PCDs.
print(scape.shape)
print(resolution * (resolution + 1) / 2)

# %%
# The :meth:`~musicflower.plotting.key_colors` function can be used to get the corresponding triangular map of colours
# as RGB in [0, 1]. These are computed by matching each PCD against templates for the 12 major and 12 minor keys and
# then interpolating between the respective colours shown in the legend above.
from musicflower.plotting import key_colors
colors = key_colors(scape)
print(colors.shape)

# %%
# Triangular Maps
# ---------------
# To handle this kind of flattened triangular maps, the :class:`~triangularmap.tmap.TMap` class from the
# `TriangularMap <https://robert-lieck.github.io/triangularmap>`_ package can be used. It takes
# a linear storage and allows for efficiently accessing its content as if it was a triangular map (see
# triangularmap documentation for more details). For multidimensional arrays, the first dimension is assumed to
# represent the triangular map.

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

# %%
# Mapping to Fourier Space
# ------------------------
# The discrete Fourier transform of a PCD contains musically relevant information. In particular, the 5th coefficient
# is strongest for distributions that correspond to diatonic scales and can therefore be associated to the tonality of a
# piece: its amplitude indicates how "strongly tonal" the piece is (e.g. atonal/12-tone pieces have a low amplitude);
# its phase maps to the circle of fifths. The :meth:`~musicflower.util.get_fourier_component` function provides
# amplitudes and phases of an array of PCDs

from musicflower.util import get_fourier_component
amplitude, phase = get_fourier_component(pcds=scape, fourier_component=5)


# %%
# Mapping to 3D Space
# -------------------
# The *amplitude* and *phase* of the Fourier component provide polar coordinates for each of the PCDs. (The phase also
# strongly correlates with the colours, even though they were computed using template matching, not Fourier components.)
# In the key scape plot above, we have two other dimensions for each PCD: *time* on the horizontal axis and the
# *duration* on the vertical axis (i.e. center and width of the respective section of the piece). Together, this can be
# used to map each PCD to a point in 3D space as follows.
#
# We use spherical or cylindrical coordinates with the *phase* as the azimuthal/horizontal angle, the *duration*
# for the radial component, and the *amplitude* for the vertical component/angle (cylinder/sphere). This is done
# by the :meth:`~musicflower.util.remap_to_xyz` function, which also provides some additional tweaks (see its
# documentation for details). Note that *time* is not explicitly represented anymore, but can be included through
# interactive animations (see below).

from musicflower.util import remap_to_xyz
x, y, z = remap_to_xyz(amplitude=amplitude, phase=phase)

# %%
# 3D Plot
# ------------------------
# This can be visualised in a 3D plot as follows

from musicflower.plotting import plot_all
plot_all(x=x, y=y, z=z, colors=colors)


# %%
# Time as Animation
# -------------------------
# We can add the time dimension using an interactive slider and/or animation. The slider represents time in a normalised
# [0, 1] interval over the piece duration. When moving the slider, a line is drawn from the top, through the triangular
# map to the point at the bottom corresponding to the current slider position. In the normal key scape plot from above,
# this would simply be a straight line from the top down to the bottom; in the 3D plot it winds through tonal space
# (also see :doc:`plot_time_traces`).

plot_all(x=x, y=y, z=z, colors=colors, do_plot_time_traces=True)
