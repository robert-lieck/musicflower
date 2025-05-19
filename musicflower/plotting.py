#  Copyright (c) 2022 Robert Lieck.

from typing import List, Union, Iterable
from numbers import Integral
from itertools import repeat

import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator

from pitchtypes import EnharmonicPitchClass

from triangularmap import TMap

from musicflower.util import get_time_traces, surface_scape_indices, assert_valid_corpus, assert_valid_xyz_col, \
    broadcast_func, iterable_or_repeat, repeat_kwargs, get_fourier_component, remap_to_xyz

# use colouring along circle of fifths (not chromatic)
pt.set_circle_of_fifths(True)


def rgba(*args):
    if len(args) == 1:
        rgba = np.asarray(args[0], dtype=float)
    else:
        rgba = np.asarray(args, dtype=float)
    if len(rgba) not in [3, 4]:
        raise ValueError("rgba values have to be specified as (an array of) 3 or 4 values")
    if np.any(rgba > 1) or np.any(rgba < 0):
        raise ValueError("rgba values should be in [0, 1]")
    rgba[:3] = rgba[:3] * 255
    rgba = [str(n) for n in rgba]
    if len(rgba) == 3:
        rgba += ["1"]
    return "rgba(" + ", ".join(rgba) + ")"


def rgba_mix(colors, weights, normalise=True):
    colors = np.asarray(colors, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("'weights' must be positive")
    if normalise:
        weights /= weights.sum()
    if len(colors.shape) != 2 or len(weights.shape) != 1:
        raise ValueError("'colors' needs to be 2D array, 'weights' 1D array")
    return rgba((colors * weights[:, None]).sum(axis=0))


def rgba_lighter(col, val):
    return rgba_mix([col, np.ones_like(col)], [1 - val, val])


def rgba_darker(col, val):
    return rgba_mix([col, np.zeros_like(col)], [1 - val, val])


def key_colors(pcds: np.ndarray, alpha=False) -> np.ndarray:
    """
    Given an array of PCDs, returns a corresponding array of RGB or RGBA colors.

    :param pcds: array of shape (..., 12) where ... can be any number of leading dimensions
    :param alpha: whether to return alpha values (i.e. RGBA colours) or just RGB
    :return: array of shape (..., 3) or (..., 4) with RGB/RGBA colors
    """
    assert pcds.shape[-1] == 12, f"last dimension of 'pcds' must be of size 12 but is of size {pcds.shape[-1]}"
    # reshape if necessary
    shape = pcds.shape
    if len(shape) > 2:
        pcds = pcds.reshape(-1, 12)
    # get scores and colours
    k = KeyEstimator()
    scores = k.get_score(pcds)
    colors = pt.key_scores_to_color(scores, circle_of_fifths=True)
    # remove alpha if requested
    if not alpha:
        colors = colors[:, :3]
    # reshape back if necessary
    if len(shape) > 2:
        colors = colors.reshape(shape[:-1] + (colors.shape[-1],))
    return colors


def plot_key_scape(corpus, show=True):
    """
    Create key scape plot(s) from corpus. If `corpus` is just a single piece (two dimensions) a single key scape plot
    is create; if `corpus` contains multiple pieces (three dimensions) multiple key scape plots in different sub-plots
    are create.

    :param corpus: array of shape ``(k, m, 12)`` or ``(m, 12)``, where ``k`` is the number of pieces and ``m``
     corresponds to the triangular map dimension.
    :param show: call pyplot.show() after creating the plots
    """
    assert_valid_corpus(corpus)
    if len(corpus.shape) == 3:
        # create multiple subplots for different pieces
        n_axes = corpus.shape[0]
        fig, axes = plt.subplots(1, n_axes, figsize=[n_axes * 6.4, 4.8])
        for c, ax in zip(corpus, axes):
            pt.scape_plot_from_array(TMap.reindex_from_top_down_to_start_end(pt.counts_to_colors(c)), ax=ax)
    else:
        # only one piece
        assert len(corpus.shape) == 2
        pt.scape_plot_from_array(TMap.reindex_from_top_down_to_start_end(pt.counts_to_colors(corpus)))
    if show:
        plt.show()


def create_fig(fig=None, trace=None, dark=True, axes_off=True, **kwargs) -> go.Figure:
    kwargs = {**dict(legend=dict(itemsizing='constant'),
                     title={
                         'x': 0.5,
                         'xanchor': 'center',
                     }), **kwargs}
    if dark:
        kwargs = {**dict(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,1)',
            plot_bgcolor='rgba(0,0,0,1)'
        ), **kwargs}
    if axes_off:
        kwargs = {**dict(scene=dict(xaxis=dict(visible=False),
                                    yaxis=dict(visible=False),
                                    zaxis=dict(visible=False))), **kwargs}
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


def grouplegend_kwargs(group, groupname, name):
    """
    Create kwargs to supply to a trace for grouping.

    :param group: an identifier for the group; used to populate value for 'legendgroup'; must be specified if
     'groupname' is specified
    :param groupname: name of the group; used to populate value for 'legendgrouptitle_text' (if 'name' is specified)
     or 'name' (if 'name' is not explicitly specified)
    :param name: name of the trace in legend; used to populate value for 'name'
    :return:
    """
    kwargs = {}
    if group is not None:
        kwargs = {**dict(legendgroup=group), **kwargs}  # trace is part of this group
    if groupname is not None:
        # this trace acts as representative of the group
        if group is None:
            raise ValueError("If 'groupname' is specified, 'group' must also be specified (and not be None).")
        # no separate name
        if name is None:
            # no separate name for this trace --> takes the group name to act as representative
            kwargs = {**dict(name=groupname), **kwargs}
        else:
            # separate name for group and this trace
            kwargs = {**dict(legendgrouptitle_text=groupname, name=name), **kwargs}
        # always show (either with separate name or only to represent group)
        kwargs = {**dict(showlegend=True), **kwargs}
    else:
        # this trace is NOT the representative of a group (but may be part of a group and also be shown sparately)
        if name is None:
            kwargs = {**dict(showlegend=False), **kwargs}  # trace is NOT shown
        else:
            kwargs = {**dict(name=name, showlegend=True), **kwargs}  # trace is shown separately
    return kwargs


def plot_points(x, y, z, colors, name=None, groupname=None, group=None, marker_kwargs=()):
    if name is True:
        name = "points"
    if name is False:
        name = None
    marker_kwargs = {**dict(color=colors, opacity=1, line=dict(width=0), size=0.3), **dict(marker_kwargs)}
    trace = go.Scatter3d(x=x, y=y, z=z,
                         mode='markers', marker=marker_kwargs,
                         **grouplegend_kwargs(group, groupname, name),
                         hoverinfo='skip')
    return trace


def plot_tip(x, y, z, colors, name=None, groupname=None, group=None):
    if name is True:
        name = "tip"
    if name is False:
        name = None
    kwargs = grouplegend_kwargs(group, groupname, name)
    if group:
        kwargs = {**dict(hovertemplate=group + "<extra></extra>"), **kwargs}
    trace = go.Scatter3d(x=x[:1], y=y[:1], z=z[:1],
                         mode='markers', marker=dict(color=colors, opacity=1, line=dict(width=0), size=5),
                         **kwargs)
    return trace


def plot_surface(x, y, z, colors, name=None, groupname=None, group=None, **kwargs):
    if name is True:
        name = "surface"
    if name is False:
        name = None
    kwargs = {**grouplegend_kwargs(group, groupname, name),
              **dict(vertexcolor=colors, opacity=0.2, hoverinfo='skip'),
              **kwargs}
    if group:
        kwargs = {**dict(hovertemplate=group + "<extra></extra>"), **kwargs}
    i, j, k = surface_scape_indices(x, -1)
    trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)
    return trace


def plot_border(x, y, z, colors, name=None, groupname=None, group=None, **kwargs):
    if name is True:
        name = "border"
    if name is False:
        name = None
    n = TMap.n_from_size(colors.shape[0])
    kwargs = {**grouplegend_kwargs(group, groupname, name),
              **dict(hoverinfo='skip', mode='lines', line_color=np.concatenate([TMap(colors).sslice[0],
                                                                                TMap(colors).eslice[n],
                                                                                TMap(colors).lslice[1]])),
              **kwargs}
    trace = go.Scatter3d(x=np.concatenate([TMap(x).sslice[0], TMap(x).eslice[n], np.flip(TMap(x).lslice[1])]),
                         y=np.concatenate([TMap(y).sslice[0], TMap(y).eslice[n], np.flip(TMap(y).lslice[1])]),
                         z=np.concatenate([TMap(z).sslice[0], TMap(z).eslice[n], np.flip(TMap(z).lslice[1])]),
                         **kwargs)
    return trace


def plot_time_traces(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, n_steps: int = None,
                     times: np.ndarray = None, group: str = None, **kwargs
                     ) -> List[go.Scatter3d]:
    """
    Plot equally spaced traces from the top to the bottom of the triangle. These can be added to a figure and
    animated with a slider using the :meth:`~musicflower.plotting.add_time_slider` function. The input arrays must
    have the same length compatible with a valid triangular map (i.e. a length of :math:`n(n+1)/2` for some integer
    :math:`n`).

    :param x: array with x-coordinates
    :param y: array with y-coordinates
    :param z: array with z-coordinates
    :param colors: array with RGB colours
    :param n_steps: see :meth:`~get_time_traces`
    :param times: see :meth:`~get_time_traces`
    :param group: optional name of a group of traces in the legend (allows for switching the time traces on/off with
     together with the other traces in that legend group)
    :return: a list of `n_steps` + 1 Plotly :class:`Scatter3d` plots
    """
    xyz_traces, colors_traces = get_time_traces(x=x, y=y, z=z, colors=colors, n_steps=n_steps, times=times)
    time_trace_plots = []
    kwargs = {**dict(showlegend=False,
                     mode='lines',
                     line_width=4,
                     hoverinfo='skip'),
              **kwargs}
    if group is not None:
        kwargs = {**dict(legendgroup=group), **kwargs}
    for xyz, colors in zip(xyz_traces, colors_traces):
        frame = go.Scatter3d(x=xyz[:, 0],
                             y=xyz[:, 1],
                             z=xyz[:, 2],
                             line_color=colors,
                             **kwargs)
        time_trace_plots.append(frame)
    return time_trace_plots


def add_time_slider(frame_traces, fig):
    frame_traces = np.array(frame_traces).T
    time_steps = frame_traces.shape[0]
    fig.update(frames=[go.Frame(data=frame_traces[i].tolist(), name=f"{i}") for i in range(time_steps)])
    fig.update_layout(sliders=[
        dict(steps=[dict(method='animate',
                         args=[[f"{k}"],
                               dict(mode='immediate',
                                    frame=dict(duration=300),
                                    transition=dict(duration=0))],
                         label=f"{k / (time_steps - 1)}")
                    for k in range(time_steps)],
             yanchor= 'top'
             )])


def plot_all(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, fig=None,
             n_time_traces: int = 200, group: str = None,
             separate_items: bool = False, groupname: str = None,
             do_plot_points: bool = True, do_plot_tip: bool = True, do_plot_surface: bool = True,
             do_plot_border: bool = True, do_plot_time_traces: bool = False,
             plot_points_kwargs=(), plot_tip_kwargs=(), plot_surface_kwargs=(),
             plot_border_kwargs=(), plot_time_traces_kwargs=(),
             key_markers=False,
             ):
    # convert default kwargs to dicts
    plot_points_kwargs = dict(plot_points_kwargs)
    plot_tip_kwargs = dict(plot_tip_kwargs)
    plot_surface_kwargs = dict(plot_surface_kwargs)
    plot_border_kwargs = dict(plot_border_kwargs)
    plot_time_traces_kwargs = dict(plot_time_traces_kwargs)
    # check inputs
    assert_valid_xyz_col(x, y, z, colors)
    # some setup
    if fig is None:
        fig = create_fig()
    multiple_pieces = len(x.shape) >= 2
    n_pieces = x.shape[0]
    # broadcast for corpus; proceed normally for single piece
    if multiple_pieces:
        # add dummy traces if time traces are plotted (workaround for Plotly bug)
        if do_plot_time_traces:
            add_dummy_traces(n_pieces, fig)
        # broadcast
        broadcast_func(plot_all, x=x, y=y, z=z, colors=colors, fig=repeat(fig),
                       n_time_traces=iterable_or_repeat(n_time_traces),
                       group=iterable_or_repeat(group, exclude=(str,)),
                       separate_items=iterable_or_repeat(separate_items),
                       groupname=iterable_or_repeat(groupname, exclude=(str,)),
                       do_plot_points=iterable_or_repeat(do_plot_points),
                       do_plot_tip=iterable_or_repeat(do_plot_tip),
                       do_plot_surface=iterable_or_repeat(do_plot_surface),
                       do_plot_border=iterable_or_repeat(do_plot_border),
                       do_plot_time_traces=repeat(False),
                       plot_points_kwargs=repeat(plot_points_kwargs),
                       plot_tip_kwargs=repeat(plot_tip_kwargs),
                       plot_surface_kwargs=repeat(plot_surface_kwargs),
                       plot_border_kwargs=repeat(plot_border_kwargs),
                       plot_time_traces_kwargs=repeat(plot_time_traces_kwargs),
                       )
    else:
        if groupname is None:
            groupname_kwargs = {}
        else:
            groupname_kwargs = dict(groupname=groupname)
        # add dummy trace if time traces are plotted (workaround for Plotly bug)
        if do_plot_time_traces:
            add_dummy_traces(1, fig)
        # plot points as scatter plot
        if do_plot_points:
            fig.add_trace(plot_points(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs, **plot_points_kwargs))
            groupname_kwargs = {}
        # plot tip as larger marker
        if do_plot_tip:
            fig.add_trace(plot_tip(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs, **plot_tip_kwargs))
            groupname_kwargs = {}
        # create surface
        if do_plot_surface:
            fig.add_trace(plot_surface(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs, **plot_surface_kwargs))
            groupname_kwargs = {}
        # trace triangle border
        if do_plot_border:
            fig.add_trace(plot_border(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs, **plot_border_kwargs))
            groupname_kwargs = {}
    if do_plot_time_traces:
        # collect time traces
        if multiple_pieces:
            time_traces = list(broadcast_func(plot_time_traces, x=x, y=y, z=z, colors=colors,
                                              group=iterable_or_repeat(group, exclude=(str,)),
                                              n_steps=iterable_or_repeat(n_time_traces),
                                              **repeat_kwargs(plot_time_traces_kwargs)))
        else:
            time_traces = [plot_time_traces(x=x, y=y, z=z, colors=colors, group=group, n_steps=n_time_traces, **plot_time_traces_kwargs)]
        # add time traces
        add_time_slider(frame_traces=time_traces, fig=fig)
    # add key markers
    if key_markers:
        add_key_markers(
            fig,
            r=1,
            parallels_kwargs=dict(opacity=0.5),
            meridians_kwargs=dict(opacity=0.5),
            label_kwargs=()
        )
    # return figure
    return fig


def toggle_group_items_separately(toggle_separately, fig):
    if toggle_separately:
        fig.update_layout(legend_groupclick='toggleitem', overwrite=True)
    else:
        fig.update_layout(legend_groupclick='togglegroup', overwrite=True)


def add_dummy_traces(n: Union[int, Iterable], fig: go.Figure) -> None:
    """
    Add empty dummy traces to the figure. This function it to fix a
    `Plotly bug <https://github.com/plotly/plotly.py/issues/3753>`_. Each animated time traces "eats" one of the
    existing traces when animation starts (automatically or when moving the slider). It starts in the order that traces
    were added. So, to not lose relevant traces, you have to add as many dummy traces as there are animated time traces.
    This has to be done FIRST, that is, BEFORE any other traces were added.

    :param n: number of dummy traces; either an integer or an iterable (which will only be used to determine the number
     of traces to add)
    :param fig: the figure where to add the dummy traces
    """
    if isinstance(n, Integral):
        it = range(n)
    else:
        it = n
    for _ in it:
        fig.add_trace(go.Scatter3d(x=[], y=[], z=[], showlegend=False))


def make_meridians(n_levels=12, offset=0, resolution=25):
    """
    Create meridians (vertical partial circles) on unit sphere.

    :param n_levels: number of meridians
    :param offset: offset of the meridians
    :param resolution: resolution of the lines
    :return: x, y, z as 2D arrays with NaNs to separate lines when flattened
    """
    theta = np.linspace(0, np.pi / 2, resolution)
    theta = np.concatenate([theta, [np.nan]])
    theta = np.repeat(theta[None], n_levels, 0)
    phi_steps = np.linspace(0, 2 * np.pi, n_levels, endpoint=False) + offset
    phi_steps = np.repeat(phi_steps[:, None], resolution + 1, 1)
    x = np.cos(phi_steps) * np.sin(theta)
    y = np.sin(phi_steps) * np.sin(theta)
    z = np.cos(theta)
    return np.stack([x, y, z])


def make_parallels(n_levels=5,
                   min_altitude=0.,
                   max_altitude=np.pi / 2,
                   resolution=100,
                   offset=0):
    """
    Create parallels (horizontal circles) on unit sphere.

    :param n_levels: number of parallels
    :param min_altitude: minimum altitude
    :param max_altitude: maximum altitude
    :param resolution: resolution of lines
    :param offset: azimuth offset
    :return: x, y, z as 2D arrays with NaNs to separate lines when flattened
    """
    phi = np.linspace(0, 2 * np.pi, resolution) + offset
    phi = np.concatenate([phi, [np.nan]])
    phi = np.repeat(phi[None], n_levels, 0)
    theta_steps = np.linspace(np.pi / 2 - min_altitude, np.pi / 2 - max_altitude, n_levels, endpoint=False)
    theta_steps = np.repeat(theta_steps[:, None], resolution + 1, 1)
    x = np.cos(phi) * np.sin(theta_steps)
    y = np.sin(phi) * np.sin(theta_steps)
    z = np.cos(theta_steps)
    return np.stack([x, y, z])


def plot_pcd_marker(pcd, labels,
                    name=None, groupname=None, group=None, r=None,
                    parallels_kwargs=(), meridians_kwargs=(), label_kwargs=()):
    pcd = np.asarray(pcd, dtype=float)
    amplitude, phase = get_fourier_component(pcds=pcd[None], fourier_component=5)
    x, y, z = remap_to_xyz(amplitude=amplitude, phase=phase)
    altitude = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))[0]
    azimuth = np.arctan2(y, x)[0]
    if r is None:
        r = np.sqrt(x ** 2 + y ** 2, z ** 2)[0]
    n_labels = len(labels)

    traces = []

    x, y, z = make_parallels(1, min_altitude=altitude) * r
    parallels_kwargs = {**dict(mode="lines", line_color="white", hoverinfo='skip'),
                        **grouplegend_kwargs(group, groupname, name),
                        **dict(parallels_kwargs)}
    traces.append(go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), **parallels_kwargs))

    x, y, z = make_meridians(n_labels, offset=azimuth) * r
    meridians_kwargs = {**dict(mode="lines", line_color="white", hoverinfo='skip'),
                        **grouplegend_kwargs(group, groupname, name),
                        **dict(meridians_kwargs)}
    traces.append(go.Scatter3d(x=x.flatten(), y=y.flatten(), z=z.flatten(), **meridians_kwargs))

    x, y, z = make_parallels(1, min_altitude=altitude, resolution=n_labels + 1, offset=azimuth) * r
    label_kwargs = {**dict(mode="markers+text", marker_color='white', marker_size=2, text=labels, hoverinfo='skip'),
                    **grouplegend_kwargs(group, groupname, name),
                    **dict(label_kwargs)}
    traces.append(go.Scatter3d(x=x[0, :n_labels], y=y[0, :n_labels], z=z[0, :n_labels], **label_kwargs))

    return traces


def add_key_markers(fig, **kwargs):
    labels = np.array([str(EnharmonicPitchClass(i)) for i in range(12)])[(np.arange(12) * 7) % 12]
    for t in (plot_pcd_marker(pcd=KeyEstimator.profiles['albrecht']['major'],
                              labels=[f"{l} major" for l in labels],
                              **kwargs) +
              plot_pcd_marker(pcd=KeyEstimator.profiles['albrecht']['minor'],
                              labels=[f"{l} minor" for l in labels],
                              **kwargs)):
        fig.add_trace(t)


def ellipse_coords(r1, r2, centre=(0, 0, 0), n=100, plane=None):
    centre = np.asarray(centre, dtype=float)
    if plane is None:
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)
    elif plane == 'xy':
        r1 = r1 * np.array([1, 0, 0])
        r2 = r2 * np.array([0, 1, 0])
    elif plane == 'xz':
        r1 = r1 * np.array([1, 0, 0])
        r2 = r2 * np.array([0, 0, 1])
    elif plane == 'yz':
        r1 = r1 * np.array([0, 1, 0])
        r2 = r2 * np.array([0, 0, 1])
    else:
        raise ValueError(f"'plane' has to be 'xy', 'xz', or 'yz' but is '{plane}'")
    assert r1.shape == (3,)
    assert r2.shape == (3,)
    assert centre.shape == (3,)
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xyz = centre[:, None] + np.cos(phi)[None, :] * r1[:, None] + np.sin(phi)[None, :] * r2[:, None]
    xyz = np.concatenate([centre[:, None], xyz], axis=1)
    x, y, z = xyz
    i = np.zeros(n)
    j = np.arange(n) + 1
    k = (np.arange(n) + 1) % n + 1
    return x, y, z, i, j, k


def ellipse_3d(r1, r2, centre=(0, 0, 0), n=100, plane=None, **kwargs):
    x, y, z, i, j, k = ellipse_coords(r1=r1, r2=r2, centre=centre, n=n, plane=plane)
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, **kwargs)
