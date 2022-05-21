#  Copyright (c) 2022 Robert Lieck.

import numpy as np

import plotly.graph_objects as go

import pitchscapes.plotting as pt
from pitchscapes.keyfinding import KeyEstimator


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
    colors = np.array([tuple(x[:4 - int(not alpha)]) for x in colors])
    return colors


def create_or_update_fig(fig=None, trace=None, dark=True, axes_off=True, **kwargs):
    kwargs = dict(legend_itemsizing='constant',
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
