"""
Time Traces
===========================

This example illustrates time traces.
"""

# %%
# Traditional Key Scape Plot
# --------------------------
# We first construct the triangular maps for a traditional key scape plot (with random colours) to illustrate time
# traces in 2D.
#
# Construct points and colours as a flat triangle

import numpy as np
n = 20
x = np.concatenate([np.arange(row + 1) - row / 2 for row in range(n)]) / (n - 1) + 0.5
y = np.zeros_like(x)
z = np.concatenate([np.ones(row + 1) * (n - 1 - row) for row in range(n)]) / (n - 1)
colors = np.random.uniform(0, 1, x.shape + (3,))

# %%
# Plot with time traces

from MusicFlower.plotting import create_fig, plot_points, plot_all
# create figure with axes etc
fig = create_fig(dark=False, axes_off=False)
# plot points (a bit larger than default)
fig.add_trace(plot_points(x=x, y=y, z=z, colors=colors, marker_kwargs=dict(size=1)))
# plot everything else (time traces are off by default)
plot_all(x=x, y=y, z=z, colors=colors, fig=fig, do_plot_time_traces=True, do_plot_points=False)
# adjust view
fig.update_layout(scene_camera=dict(eye=dict(x=0.2, y=-2, z=0.2), center=dict(x=0, y=0, z=0)))
fig

# %%
# 3D Space
# --------
# This surface may be embedded in 3D space in a different way, but the time traces still work the same way.

x_ = np.cos(10 * x) * (x + z ** 2)
y_ = np.sin(10 * x) * (x + z ** 2)
z_ = z
plot_all(x=x_, y=y_, z=z_, colors=colors, do_plot_time_traces=True)
