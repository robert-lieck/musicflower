#  Copyright (c) 2022 Robert Lieck.

import numpy as np
from scipy import interpolate

import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator

from TriangularMap import TMap

# use colouring along circle of fifths (not chromatic)
pt.set_circle_of_fifths(True)


def key_colors(pcds: np.ndarray, alpha=False) -> np.ndarray:
    """
    Given an array of Kx12 PCDs, returns a Kx4 array of RGBA colors.

    :param pcds: array of Kx12 PCDs
    :param alpha: whether to return alpha values or just RGB
    :return: Kx4 array of RGBA colors
    """
    k = KeyEstimator()
    scores = k.get_score(pcds)
    colors = pt.key_scores_to_color(scores, circle_of_fifths=True)
    if not alpha:
        colors = colors[:, :3]
    return colors


def plot_key_scape(c):
    pt.scape_plot_from_array(TMap.reindex_start_end_from_top_down(pt.counts_to_colors(c)))
    plt.show()


def create_or_update_fig(fig=None, trace=None, dark=True, axes_off=True, **kwargs):
    kwargs = dict(legend=dict(itemsizing='constant'),
                  title={
                      'x': 0.5,
                      'xanchor': 'center',
                  }) | kwargs
    if dark:
        kwargs = dict(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,1)',
            plot_bgcolor='rgba(0,0,0,1)'

        ) | kwargs
    if axes_off:
        kwargs = dict(scene=dict(xaxis=dict(visible=False),
                                 yaxis=dict(visible=False),
                                 zaxis=dict(visible=False))) | kwargs
    if fig is None:
        if trace is None:
            fig = go.Figure(layout=kwargs)
        else:
            fig = go.Figure(data=[trace], layout=kwargs)
    else:
        if trace is not None:
            fig.add_trace(trace)
        if kwargs:
            fig.update_layout(**kwargs)
    return fig


def trisurf(triangles, decimals=None):
    """
    Convert a Nx3x3 array of N triangles in 3D space, defined via their corners, into x, y, z coordinate arrays, and
    i, j, k index arrays as used by Plotly's Mesh3d function. Additionally, a return index r is return that can be used
    to reconstruct information associated  with the original points.

    :param triangles: Nx3x3 array of triangle vertices in 3D space (last dimension is x-y-z coordinate)
    :param decimals: decimals to use for rounding using numpy.around to improve point identification
    (default: None i.e. do not round)
    :return: x, y, z, i, j, k, r
    """
    # asser shape is OK
    assert triangles.shape[1:] == (3, 3), f"'triangles' has to have shape Nx3x3 but has shape {triangles.shape}"
    # round if requested
    if decimals is not None:
        triangles = np.around(triangles, decimals=decimals)
    # flatten to array of 3D points
    triangle_points = triangles.reshape(-1, 3)
    # get unique points (this also sorts!)
    unique_points, return_index = np.unique(triangle_points, axis=0, return_index=True)
    # convert to array of tuples for index finding
    triangle_points_as_tuples = np.array([tuple(x) for x in triangle_points], dtype="f,f,f")
    unique_points_as_tuples = np.array([tuple(x) for x in unique_points], dtype="f,f,f")
    # find indices of triangle points (relies on sorted unique points)
    triangle_point_indices = np.searchsorted(unique_points_as_tuples, triangle_points_as_tuples)
    # reshape to array of 3D indices
    triangle_point_indices = triangle_point_indices.reshape(-1, 3)
    # return separate arrays
    return (unique_points[:, 0], unique_points[:, 1], unique_points[:, 2],
            triangle_point_indices[:, 0], triangle_point_indices[:, 1], triangle_point_indices[:, 2],
            return_index)


def surface_scape_indices(n):
    """
    Return i, j, k point indices for triangles creating a scape surface of size n

    :param n: size
    :return: i, j, k
    """
    index_map = TMap(np.arange(TMap.size1d_from_n(n)))
    i, j, k = [], [], []
    for d in range(n - 1):
        a_slice = index_map.dslice(d)
        b_slice = index_map.dslice(d + 1)
        # triangles with tip up
        i.append(a_slice)
        j.append(b_slice[1:])
        k.append(b_slice[:-1])
        # triangles with tip down
        if d < n - 2:
            c_slice = index_map.dslice(d + 2)
            i.append(b_slice[:-1])
            j.append(b_slice[1:])
            k.append(c_slice[1:-1])
    return np.concatenate(i), np.concatenate(j), np.concatenate(k)


def time_traces(x, y, z, colors, m):
    xyz = TMap(np.concatenate([x[:, None], y[:, None], z[:, None]], axis=-1))
    colors = TMap(colors)
    n = colors.n
    xyz_out = np.zeros((m, n, 3))
    colors_out = np.ones((m, n, 3))
    for depth in range(n):
        for arr, arr_out in [(xyz, xyz_out), (colors, colors_out)]:
            for i in range(3):
                if depth == 0:
                    arr_out[:, depth, i] = arr.dslice(depth)[0, i]
                else:
                    arr_out[:, depth, i] = interpolate.interp1d(np.linspace(0, 1, depth + 1),
                                                                arr.dslice(depth)[:, i],
                                                                copy=False)(np.linspace(0, 1, m))
    return xyz_out, colors_out


def grouplegend_kwargs(group, groupname, name):
    kwargs = {}
    if group is not None:
        kwargs |= dict(legendgroup=group)  # trace is part of this group
    if groupname is not None:
        # this trace acts as representative of the group
        if group is None:
            raise ValueError("If 'groupname' is specified, 'group' must also be specified (and not be None).")
        # no separate name
        if name is None:
            # no separate name for this trace --> takes the group name to act as representative
            kwargs |= dict(name=groupname)
        else:
            # separate name for group and this trace
            kwargs |= dict(legendgrouptitle_text=groupname, name=name)
        # always show (either with separate name or only to represent group)
        kwargs |= dict(showlegend=True)
    else:
        # this trace is NOT the representative of a group (but may be part of a group and also be shown sparately)
        if name is None:
            kwargs |= dict(showlegend=False)  # trace is NOT shown
        else:
            kwargs |= dict(name=name, showlegend=True)  # trace is shown separately
    return kwargs


def plot_points(x, y, z, colors, name=None, groupname=None, group=None, fig=None):
    trace = go.Scatter3d(x=x, y=y, z=z,
                         mode='markers', marker=dict(color=colors, opacity=1, line=dict(width=0), size=0.3),
                         **grouplegend_kwargs(group, groupname, name),
                         hoverinfo='skip',
                         # legendgroup=label,
                         # **(dict(
                         #     # separate labels for group and this trace
                         #     legendgrouptitle_text=label, name="points",
                         # ) if separate_items else dict(
                         #     # use this trace to represent whole group
                         #     name=label
                         # )),
                         # showlegend=True,  # always show (as separate trace or to represent group)
                         )
    if fig is not None:
        create_or_update_fig(fig=fig, trace=trace)
    return trace


def plot_tip(x, y, z, colors, name=None, groupname=None, group=None, fig=None):
    kwargs = grouplegend_kwargs(group, groupname, name)
    if group:
        kwargs |= dict(hovertemplate=group + "<extra></extra>",)
    trace = go.Scatter3d(x=x[:1], y=y[:1], z=z[:1],
                         mode='markers', marker=dict(color=colors, opacity=1, line=dict(width=0), size=5),
                         **kwargs)
    if fig is not None:
        create_or_update_fig(fig=fig, trace=trace)
    return trace


def plot_surface(x, y, z, colors, name=None, groupname=None, group=None, fig=None):
    kwargs = grouplegend_kwargs(group, groupname, name)
    if group:
        kwargs |= dict(hovertemplate=group + "<extra></extra>", )
    i, j, k = surface_scape_indices(TMap.n_from_size1d(colors.shape[0]))
    trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=colors, opacity=0.2, **kwargs)
    if fig is not None:
        create_or_update_fig(fig=fig, trace=trace)
    return trace


def plot_border(x, y, z, colors, name=None, groupname=None, group=None, fig=None):
    n = TMap.n_from_size1d(colors.shape[0])
    trace = go.Scatter3d(x=np.concatenate([TMap(x).sslice[0], TMap(x).eslice[n], TMap(x).lslice(1)]),
                         y=np.concatenate([TMap(y).sslice[0], TMap(y).eslice[n], TMap(y).lslice(1)]),
                         z=np.concatenate([TMap(z).sslice[0], TMap(z).eslice[n], TMap(z).lslice(1)]),
                         mode='lines', line=dict(color=np.concatenate([TMap(colors).sslice[0],
                                                                       TMap(colors).eslice[n],
                                                                       TMap(colors).lslice(1)])),
                         **grouplegend_kwargs(group, groupname, name),
                         hoverinfo='skip')
    if fig is not None:
        create_or_update_fig(fig=fig, trace=trace)
    return trace


def plot_time_traces(x, y, z, colors, n_steps, group=None):
    xyz_traces, colors_traces = time_traces(x=x, y=y, z=z, colors=colors, m=n_steps)
    time_trace_plots = []
    kwargs = dict(showlegend=False)
    if group is not None:
        kwargs |= dict(legendgroup=group)
    for i in range(n_steps):
        frame = go.Scatter3d(x=xyz_traces[i, :, 0],
                             y=xyz_traces[i, :, 1],
                             z=xyz_traces[i, :, 2],
                             mode='lines', line=dict(color=colors_traces[i, :], width=4),
                             hoverinfo='skip')
        time_trace_plots.append(frame)
    return time_trace_plots


def add_time_slider(frame_traces, fig):
    frame_traces = np.array(frame_traces).T
    time_steps = frame_traces.shape[0]
    fig.update(frames=[go.Frame(data=frame_traces[i].tolist(), name=f"{i}") for i in range(time_steps)])
    fig.update_layout(sliders=[dict(steps=[dict(method='animate',
                                                args=[[f"{k}"],
                                                      dict(mode='immediate',
                                                           frame=dict(duration=300),
                                                           transition=dict(duration=0))],
                                                label=f"{k / time_steps}")
                                           for k in range(time_steps)])])


def toggle_group_items_separately(toggle_separately, fig):
    if toggle_separately:
        fig.update_layout(legend_groupclick='toggleitem', overwrite=True)
    else:
        fig.update_layout(legend_groupclick='togglegroup', overwrite=True)


def main():
    pass


if __name__ == "__main__":
    main()
