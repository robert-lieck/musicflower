#  Copyright (c) 2022 Robert Lieck.

from typing import List, Union, Iterable
from numbers import Integral
from itertools import repeat

import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator

from triangularmap import TMap

from musicflower.util import get_time_traces, surface_scape_indices, assert_valid_corpus, assert_valid_xyz_col, \
    broadcast_func, iterable_or_repeat

# use colouring along circle of fifths (not chromatic)
pt.set_circle_of_fifths(True)


def key_colors(pcds: np.ndarray, alpha=False) -> np.ndarray:
    """
    Given an array of PCDs, returns a corresponding array of RGB or RGBA colors.

    :param pcds: array of shape (..., 12) where can be any number of leading dimensions
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
            pt.scape_plot_from_array(TMap.reindex_start_end_from_top_down(pt.counts_to_colors(c)), ax=ax)
    else:
        # only one piece
        assert len(corpus.shape) == 2
        pt.scape_plot_from_array(TMap.reindex_start_end_from_top_down(pt.counts_to_colors(corpus)))
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


def plot_surface(x, y, z, colors, name=None, groupname=None, group=None):
    if name is True:
        name = "surface"
    if name is False:
        name = None
    kwargs = grouplegend_kwargs(group, groupname, name)
    if group:
        kwargs = {**dict(hovertemplate=group + "<extra></extra>"), **kwargs}
    i, j, k = surface_scape_indices(x, -1)
    trace = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, vertexcolor=colors, opacity=0.2, **kwargs)
    return trace


def plot_border(x, y, z, colors, name=None, groupname=None, group=None):
    if name is True:
        name = "border"
    if name is False:
        name = None
    n = TMap.n_from_size1d(colors.shape[0])
    trace = go.Scatter3d(x=np.concatenate([TMap(x).sslice[0], TMap(x).eslice[n], np.flip(TMap(x).lslice(1))]),
                         y=np.concatenate([TMap(y).sslice[0], TMap(y).eslice[n], np.flip(TMap(y).lslice(1))]),
                         z=np.concatenate([TMap(z).sslice[0], TMap(z).eslice[n], np.flip(TMap(z).lslice(1))]),
                         mode='lines', line=dict(color=np.concatenate([TMap(colors).sslice[0],
                                                                       TMap(colors).eslice[n],
                                                                       TMap(colors).lslice(1)])),
                         **grouplegend_kwargs(group, groupname, name),
                         hoverinfo='skip')
    return trace


def plot_time_traces(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, n_steps: int, group: str = None
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
    :param n_steps: number time intervals; `n_steps` + 1 traces are created (incl. time zero and one)
    :param group: optional name of a group of traces in the legend (allows for switching the time traces on/off with
     together with the other traces in that legend group)
    :return: a list of `n_steps` + 1 Plotly :class:`Scatter3d` plots
    """
    xyz_traces, colors_traces = get_time_traces(x=x, y=y, z=z, colors=colors, n_steps=n_steps)
    time_trace_plots = []
    kwargs = dict(showlegend=False)
    if group is not None:
        kwargs = {**dict(legendgroup=group), **kwargs}
    for i in range(n_steps + 1):
        frame = go.Scatter3d(x=xyz_traces[i, :, 0],
                             y=xyz_traces[i, :, 1],
                             z=xyz_traces[i, :, 2],
                             mode='lines', line=dict(color=colors_traces[i, :], width=4),
                             hoverinfo='skip',
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
             do_plot_border: bool = True, do_plot_time_traces: bool = False):
    assert_valid_xyz_col(x, y, z, colors)
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
                       do_plot_time_traces=repeat(False))
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
            fig.add_trace(plot_points(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs))
            groupname_kwargs = {}
        # plot tip as larger marker
        if do_plot_tip:
            fig.add_trace(plot_tip(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs))
            groupname_kwargs = {}
        # create surface
        if do_plot_surface:
            fig.add_trace(plot_surface(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs))
            groupname_kwargs = {}
        # trace triangle border
        if do_plot_border:
            fig.add_trace(plot_border(x=x, y=y, z=z, colors=colors, name=separate_items, group=group, **groupname_kwargs))
            groupname_kwargs = {}
    if do_plot_time_traces:
        # collect time traces
        if multiple_pieces:
            time_traces = list(broadcast_func(plot_time_traces, x=x, y=y, z=z, colors=colors,
                                              group=iterable_or_repeat(group, exclude=(str,)),
                                              n_steps=iterable_or_repeat(n_time_traces)))
        else:
            time_traces = [plot_time_traces(x=x, y=y, z=z, colors=colors, group=group, n_steps=n_time_traces)]
        # add time traces
        add_time_slider(frame_traces=time_traces, fig=fig)
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


def main():
    pass


if __name__ == "__main__":
    main()
