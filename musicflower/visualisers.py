#  Copyright (c) 2025 Robert Lieck.

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from pitchtypes import EnharmonicPitchClass, SpelledPitchClass

from musicflower.util import transpose_profiles, rad_to_deg, tonal_modulo, remap_to_xyz, major_minor_profiles
from musicflower.features import fourier_features, use_chroma_features, use_fourier_features, use_chroma_scape_features, use_fourier_scape_features
from musicflower.plotting import (rgba, rgba_lighter, rgba_mix, plot_all, key_colors, add_key_markers, plot_time_traces,
                                  create_fig, ellipse_3d)
from musicflower.webapp import WebApp


def waveform_visualiser(*, features, position, app=WebApp, update=True, **kwargs):
    features = app.check_features(features[0], n=2)
    if update:
        if position is None:
            y, sr = features
            waveform = px.line(y)
            fig = go.Figure(data=[
                waveform.data[0],
                go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, dash="dash", color="red"))
            ], layout=waveform.layout)
        else:
            x_max = (len(features[0]) - 1)
            pos = position * x_max
            return [{}, dict(x=[pos, pos], y=[-1, 1])]
    else:
        y, sr = features
        fig = px.line(y)
        fig.add_vline(position * (y.shape[0] - 1), line_width=3, line_dash="dash", line_color="red", opacity=0.9)
    return app.update_figure_layout(fig, **kwargs)


def heatmap_visualiser(*, features, position, app=WebApp, update=False, express=True, **kwargs):
    features = app.check_features(features)
    if update:
        if position is None:
            heatmap = px.imshow(features.T, origin='lower', aspect="auto")
            fig = go.Figure(data=[
                heatmap.data[0],
                go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, dash="dash", color="red"))
            ], layout=heatmap.layout)
        else:
            x_max, y_max = np.array(features.shape) - 1
            pos = position * x_max
            return [{}, dict(x=[pos, pos], y=[0, y_max])]
    else:
        x_max, y_max = np.array(features.shape) - 1
        pos = position * x_max
        if express:
            fig = px.imshow(features.T, origin='lower', aspect="auto")
            fig.add_vline(pos, line_width=3, line_dash="dash", line_color="red", opacity=0.9)
        else:
            heatmap = px.imshow(features.T, origin='lower', aspect="auto")
            fig = go.Figure(data=[
                heatmap.data[0],
                go.Scatter(x=[pos, pos], y=[0, y_max], mode='lines', line=dict(width=3, dash="dash", color="red"))
            ], layout=heatmap.layout)
    return app.update_figure_layout(fig, **kwargs)


def advanced_chroma_visualiser_fast(*, features, position, app=WebApp, **kwargs):
    features = app.check_features(features)
    x_max = (features.shape[0] - 1)
    if position is None:
        heatmap = px.imshow(features.T, origin='lower', aspect="auto")
        fig = go.Figure(data=[
            heatmap.data[0],
            go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, dash="dash", color="red"))
        ], layout=heatmap.layout)
        return app.update_figure_layout(fig, **kwargs)
    else:
        pos = position * x_max
        return [{}, dict(x=[pos, pos], y=[0, 11])]


def single_fourier(*, features, position, app=WebApp, component, **kwargs):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    fig = px.line_polar(r=features[0, :, component], theta=features[1, :, component])
    idx = app.position_idx(position, n=features.shape[1])
    fig.add_trace(go.Scatterpolar(
        r=features[0, idx:idx + 1, component],
        theta=features[1, idx:idx + 1, component],
        mode='markers',
        marker=dict(size=10, color='red')
    ))
    return app.update_figure_layout(fig, **kwargs)


def fourier_visualiser(*, features, position, app=WebApp, binary_profiles=False, incl=None, **kwargs):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    labels = [f"{nth} Coefficient" for nth in ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th']]
    specs = [[{'type': 'polar'} for _ in range(3)] for _ in range(2)]
    specs[1][2] = {}
    fig = make_subplots(rows=2, cols=3, start_cell="top-left", specs=specs,
                        # subplot_titles=labels
                        )
    idx = app.position_idx(position, n=features.shape[1])
    # add Fourier components 1–5
    for (component, row, col), l in zip([
        (0, 0, 0),
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (4, 2, 1),
        (5, 2, 2),
        (6, 2, 3),
    ], labels):
        # don't plot real-valued 0th and 6th component as polar plots
        if component in [0, 6] or (incl is not None and component not in incl):
            continue
        # plot trace
        fig.add_trace(go.Scatterpolar(
            hoverinfo='skip',
            r=features[0, :, component],
            theta=features[1, :, component],
            mode='lines',
            name=l,
        ), row=row, col=col)
        # plot marker
        fig.add_trace(go.Scatterpolar(
            hoverinfo='skip',
            r=features[0, idx:idx + 1, component],
            theta=features[1, idx:idx + 1, component],
            mode='markers',
            marker=dict(size=10, color='red'),
            showlegend=False,
        ), row=row, col=col)
        # plot landmarks
        def get_landmarks(profile, component):
            profiles = transpose_profiles(np.array(profile))  # (trans, pitch)
            fourier_profiles = fourier_features(features=[profiles])  # (mag/phase, trans, component)
            return fourier_profiles[..., component]  # (mag/phase, trans)
        angularaxis = {}
        fourier_profiles = None
        pitch_names = [str(EnharmonicPitchClass(n)).replace('#', '♯').replace('b', '♭') for n in range(12)]
        if component == 1:
            fourier_profiles = get_landmarks([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], component)
            angularaxis['ticktext'] = np.array([f"{n}" for n in pitch_names])
        elif component == 2:
            fourier_profiles = get_landmarks([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], component)[:, :6]
            angularaxis['ticktext'] = [f"{pitch_names[i]}T" for i in range(6)]
        elif component == 3:
            fourier_profiles = get_landmarks([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], component)[:, :4]
            angularaxis['ticktext'] = [f"{pitch_names[i]}+" for i in range(4)]
        elif component == 4:
            fourier_profiles = get_landmarks([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], component)[:, :3]
            angularaxis['ticktext'] = [f"{pitch_names[i]}º" for i in range(3)]
        elif component == 5:
            if binary_profiles:
                # key signatures represented by binary profiles
                fourier_profiles = get_landmarks([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], component)
                angularaxis['ticktext'] = tonal_modulo(np.array(
                    ["♮", "♯", "2♯", "3♯", "4♯", "5♯", "6♯", "5♭", "4♭", "3♭", "2♭", "♭"]
                ))
            else:
                # major/minor keys represented by Albrecht profiles
                fourier_profiles = fourier_features(features=[major_minor_profiles])  # (mag/phase, mode, trans, component)
                # concatenate major/minor mode
                fourier_profiles = np.concatenate([fourier_profiles[:, 0, :, component],
                                                   fourier_profiles[:, 1, :, component]], axis=1)
                # get ticks
                angularaxis['ticktext'] = np.array([f"{n}ma" for n in pitch_names] + [f"{n}mi" for n in pitch_names])
        if fourier_profiles is not None:
            angularaxis['tickmode'] = 'array'
            angularaxis['tickvals'] = np.mod(fourier_profiles[1, :] * rad_to_deg, 360)
            fig.add_trace(go.Scatterpolar(
                hoverinfo='skip',
                r=fourier_profiles[0, :],
                theta=fourier_profiles[1, :] * rad_to_deg,
                mode='markers',
                marker=dict(size=4, color='black'),
                showlegend=False,
            ), row=row, col=col)
        # adjust axes
        fig.update_polars(
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks=''),
            angularaxis=angularaxis,
            angularaxis_direction='clockwise',
            row=row, col=col
        )
    # add 0th and 6th component
    x = []
    y = []
    if incl is None or 0 in incl:
        x.append(labels[0])
        y.append(features[0, idx, 0] * np.cos(features[1, idx, 0]))
    if incl is None or 6 in incl:
        x.append(labels[6])
        y.append(features[0, idx, 6] * np.cos(features[1, idx, 6]))
    if x and y:
        fig.add_trace(go.Bar(x=x, y=y, name="0th/6th Coefficient",
                             hoverinfo='skip',
            # x=[labels[0], labels[6]],
            # y=features[0, idx, (0, 6)] * np.cos(features[1, idx, (0, 6)]),
        ), row=2, col=3)
        fig.update_yaxes(range=[-1.1, 1.1], row=2, col=3)
    kwargs = {**dict(margin=dict(t=30, b=10, l=0, r=0),), **kwargs}
    return app.update_figure_layout(fig, **kwargs)


def use_fourier_visualiser(app):
    use_fourier_features(app, ignore_existing=True)
    app.register_visualiser('Fourier Coefficients',
                            ['fourier-features'],
                            fourier_visualiser)


def circle_of_fifths_visualiser(*, features, position, app=WebApp, ticks="binary", **kwargs):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    idx = app.position_idx(position, n=features.shape[1])
    component = 5
    fig = go.Figure()
    # plot trace
    fig.add_trace(go.Scatterpolar(
        hoverinfo='skip',
        name="",
        showlegend=False,
        r=features[0, :, component],
        theta=features[1, :, component],
        mode='lines'))
    # plot marker
    fig.add_trace(go.Scatterpolar(
        hoverinfo='skip',
        r=features[0, idx:idx + 1, component],
        theta=features[1, idx:idx + 1, component],
        mode='markers',
        marker=dict(size=10, color='red'),
        showlegend=False))
    # default: don't change angular ticks
    angularaxis = {}
    # plot key signatures represented by binary profiles
    accidentals = tonal_modulo(np.array(["♮", "♯", "2♯", "3♯", "4♯", "5♯", "6♯", "5♭", "4♭", "3♭", "2♭", "♭"]))
    profiles = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7
    profiles = transpose_profiles(profiles)  # (trans, pitch)
    fourier_profiles = fourier_features(features=[profiles])  # (mag/phase, trans, component)
    fig.add_trace(go.Scatterpolar(
        hovertemplate="%{text}",
        text=accidentals,
        name="key signature",
        r=fourier_profiles[0, :, component],
        theta=fourier_profiles[1, :, component] * rad_to_deg,
        mode='markers',
        marker=dict(size=2, color='black'),
        showlegend=False))
    # set ticks
    if ticks == "binary":
        angularaxis = dict(
            tickmode='array',
            tickvals=np.mod(fourier_profiles[1, :, component] * rad_to_deg, 360),
            ticktext=accidentals,
        )
    # major/minor keys represented by Albrecht profiles
    pitch_names = [str(EnharmonicPitchClass(n)) for n in range(12)]
    ma_mi_labels = np.array([[f"{n}ma", f"{n}mi"] for n in pitch_names]).T.flatten()
    # profiles = KeyEstimator.profiles['albrecht']
    # profiles = np.array([profiles['major'], profiles['minor']]).T  # (pitch, mode)
    # profiles = transpose_profiles(profiles)  # (trans, pitch, mode)
    # profiles = np.moveaxis(profiles, 2, 0)  # (mode, trans, pitch)
    fourier_profiles = fourier_features(features=[major_minor_profiles])  # (mag/phase, mode, trans, component)
    fig.add_trace(go.Scatterpolar(
        hovertemplate="%{text}",
        text=ma_mi_labels,
        name="key",
        r=fourier_profiles[0, :, :, component].flatten(),
        theta=fourier_profiles[1, :, :, component].flatten() * rad_to_deg,
        mode='markers',
        marker=dict(size=5, color='black'),
        showlegend=False))
    # set ticks
    if ticks == "albrecht":
        ma_mi_phase = np.concatenate([fourier_profiles[1, 0, :, component], fourier_profiles[1, 1, :, component]])
        angularaxis = dict(
            tickmode='array',
            tickvals=np.mod(ma_mi_phase * rad_to_deg, 360),
            ticktext=ma_mi_labels,
        )
    # adjust axes
    fig.update_polars(
        radialaxis=dict(range=[0, 1], showticklabels=False, ticks=''),
        angularaxis=angularaxis,
    )
    return app.update_figure_layout(fig, **kwargs)


def tonnetz_visualiser(*, features, position, app=WebApp, unicode=True, **kwargs):
    features = app.check_features(features, asfarray=False)
    pos_idx = app.position_idx(position, features=features)
    features = np.array(features[pos_idx])
    if features.max() > 0:
        features /= features.max()

    fig = go.Figure(layout=dict(
        xaxis=dict(zeroline=False, showgrid=False, visible=False),
        yaxis=dict(zeroline=False, showgrid=False, visible=False, scaleanchor="x", scaleratio=1),
        template='simple_white',
    ))
    lof = np.arange(-6, 7)
    lot = np.arange(-4, 5)
    ex_5 = 1
    ey_5 = 0
    ex_mi3 = np.cos(np.pi / 3)
    ey_mi3 = np.sin(np.pi / 3)
    ex_ma3 = ex_5 - ex_mi3
    ey_ma3 = ey_5 - ey_mi3
    fifths_per_mi3 = -3

    pc_size = 0.4
    pc_font_size = 16
    pc_neural_col = [0.9, 0.9, 0.9]
    pc_on_col = [0.9, 0, 0]
    chord_size = 0.1
    chord_font_size = 9
    major_col = np.array([0.8, 0.1, 0.1])
    minor_col = np.array([0.1, 0.1, 0.8])
    chord_lightning = 0.7
    chord_label_lightning = 0.8
    neo_P_col = rgba(0.8, 0, 0)
    neo_R_col = rgba(0.8, 0.8, 0)
    neo_L_col = rgba(0, 0, 1)

    # iterate along minor third direction
    def third_iterator():
        for i, n_mi3 in enumerate(lot):
            x = ex_5 * lof + ex_mi3 * n_mi3
            y = ey_5 * lof + ey_mi3 * n_mi3
            yield x, y, n_mi3, (i == 0, i == len(lot) - 1)

    # iterate over entire grid (minor third and fifth direction)
    def grid_iterator():
        for x, y, _, lim in third_iterator():
            for i, (x_, y_) in enumerate(zip(x, y)):
                yield x_, y_, lim + (i == 0, i == len(x) - 1)

    # position of chords for shift by n_mi3 minor thirds
    def chord_pos(n_mi3):
        x_mi3 = x[:-1] + (ex_5 + ex_mi3) / 3
        y_mi3 = y[:-1] + (ey_5 + ey_mi3) / 3
        x_ma3 = x[:-1] + (ex_5 + ex_ma3) / 3
        y_ma3 = y[:-1] + (ey_5 + ey_ma3) / 3
        return x_mi3, y_mi3, x_ma3, y_ma3

    # create circle as a filled trace
    def circle_trace(x, y, r, n=100, **kwargs):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=True)
        x_off = r * np.cos(phi)
        y_off = r * np.sin(phi)
        if isinstance(x, np.ndarray):
            x = np.concatenate([x_off[None, :] + x[:, None], np.repeat(None, len(x))[:, None]], axis=1).flatten()
            y = np.concatenate([y_off[None, :] + y[:, None], np.repeat(None, len(y))[:, None]], axis=1).flatten()
        else:
            x = x_off + x
            y = y_off + y
        kwargs = dict(fill="toself",
                      hoverinfo='skip',
                      showlegend=False) | kwargs
        return go.Scatter(x=x, y=y, **kwargs)

    # add chords
    for x, y, lim in grid_iterator():
        circle_kwargs = dict(mode="none",
                      fill="toself",
                      hoverinfo='skip',
                      showlegend=False)
        # triangle for major triads
        if not (lim[0] or lim[3]):
            fig.add_trace(go.Scatter(
                x=[x, x + ex_ma3, x + ex_5, x],
                y=[y, y + ey_ma3, y + ey_5, y],
                fillcolor=rgba_lighter(major_col, chord_lightning),
                **circle_kwargs,
            ))
        # triangle for minor triads
        if not (lim[1] or lim[3]):
            fig.add_trace(go.Scatter(
                x=[x, x + ex_mi3, x + ex_5, x],
                y=[y, y + ey_mi3, y + ey_5, y],
                fillcolor=rgba_lighter(minor_col, chord_lightning),
                **circle_kwargs,
            ))
    # neo-Riemannian relations
    set_p = True
    set_r = True
    set_l = True
    for x, y, n_mi3, lim in third_iterator():
        x_mi3, y_mi3, x_ma3, y_ma3 = chord_pos(n_mi3)
        separator = np.repeat(None, len(x_ma3))
        if not (lim[0] or lim[1]):
            fig.add_trace(go.Scatter(
                x=np.stack([x_ma3, x_mi3, separator], axis=-1).flatten(),
                y=np.stack([y_ma3, y_mi3, separator], axis=-1).flatten(),
                line=dict(color=neo_P_col), legendgroup="neo", name=f"P", legendgrouptitle_text="Neo-Riem.",
                showlegend=set_p, hoverinfo='skip',))
            set_p = False
        if not lim[0]:
            fig.add_trace(go.Scatter(
                x=np.stack([x_ma3, x_mi3 - ex_mi3, separator], axis=-1).flatten(),
                y=np.stack([y_ma3, y_mi3 - ey_mi3, separator], axis=-1).flatten(),
                line=dict(color=neo_R_col), legendgroup="neo", name=f"R", legendgrouptitle_text="Neo-Riem.",
                showlegend=set_r, hoverinfo='skip',))
            set_r = False
            fig.add_trace(go.Scatter(
                x=np.stack([x_ma3[:-1], x_mi3[:-1] + ex_5 - ex_mi3, separator[:-1]], axis=-1).flatten(),
                y=np.stack([y_ma3[:-1], y_mi3[:-1] + ey_5 - ey_mi3, separator[:-1]], axis=-1).flatten(),
                line=dict(color=neo_L_col), legendgroup="neo", name=f"L", legendgrouptitle_text="Neo-Riem.",
                showlegend=set_l, hoverinfo='skip',))
            set_l = False
    # add labels
    for x, y, n_mi3, lim in third_iterator():
        x_mi3, y_mi3, x_ma3, y_ma3 = chord_pos(n_mi3)
        pc_labels = [str(SpelledPitchClass(value=fifths + n_mi3 * fifths_per_mi3)) for fifths in lof]
        if unicode:
            pc_labels = [l.replace('#', '♯').replace('b', '♭') for l in pc_labels]
        if n_mi3 > 0:
            pc_labels_syn = [l + "'" * n_mi3 for l in pc_labels]
        elif n_mi3 < 0:
            pc_labels_syn = [l + "," * (-n_mi3) for l in pc_labels]
        else:
            pc_labels_syn = pc_labels
        pc_idx = [int(SpelledPitchClass(value=fifths + n_mi3 * fifths_per_mi3).convert_to(EnharmonicPitchClass)) % 12 for fifths in lof]
        label_kwargs = dict(mode="text",
                            hoverinfo='skip',
                            showlegend=False)
        # create circle for pitch classes
        for pc_x, pc_y, i in zip(x, y, pc_idx):
            w = features[i]
            fig.add_trace(circle_trace(pc_x, pc_y, pc_size,
                                       fillcolor=rgba_mix([pc_neural_col, pc_on_col], [1 - w, w]),
                                       line_color=rgba(0.1, 0.1, 0.1)))
        # create labels for pitch classes
        fig.add_trace(go.Scatter(x=x, y=y, text=pc_labels_syn, textfont=dict(size=pc_font_size), **label_kwargs))
        # circles/lables for major/minor triads
        circle_kwargs = dict(mode="none",
                             fill="toself",
                             hoverinfo='skip',
                             showlegend=False,
                             r=chord_size)
        # major
        if not lim[0]:
            fig.add_trace(circle_trace(x_ma3, y_ma3, fillcolor=rgba_lighter(major_col, chord_label_lightning), **circle_kwargs))
            fig.add_trace(go.Scatter(
                x=x[:-1] + (ex_5 + ex_ma3) / 3,
                y=y[:-1] + (ey_5 + ey_ma3) / 3,
                text=[f"{l}ma" for l in pc_labels[:-1]],
                textfont=dict(size=chord_font_size),
                **label_kwargs
            ))
        # minor
        if not lim[1]:
            fig.add_trace(circle_trace(x_mi3, y_mi3, fillcolor=rgba_lighter(minor_col, chord_label_lightning), **circle_kwargs))
            fig.add_trace(go.Scatter(
                x=x[:-1] + (ex_5 + ex_mi3) / 3,
                y=y[:-1] + (ey_5 + ey_mi3) / 3,
                text=[f"{l}mi" for l in pc_labels[:-1]],
                textfont=dict(size=chord_font_size),
                **label_kwargs
            ))
    # indicate unitcell
    # uc_5 = np.array([-3, 0, 3, 0, -3])
    # uc_mi3 = np.array([1, 2, -1, -2, 1])
    uc_5 = np.array([-6, 6, np.nan, -6, 6, np.nan, -6, -2, np.nan, -6, 2, np.nan, -2, 6, np.nan, 2, 6])
    uc_mi3 = np.array([0, 4, np.nan, -4, 0, np.nan, 0, -4, np.nan, 4, -4, np.nan, 4, -4, np.nan, 4, 0])
    fig.add_trace(go.Scatter(
        x=uc_5 * ex_5 + uc_mi3 * ex_mi3,
        y=uc_5 * ey_5 + uc_mi3 * ey_mi3,
        name="unit cell",
        hoverinfo='skip',
        mode="lines",
        line=dict(width=2, color=rgba(0, 0, 0, 0.5), dash='dot')
    ))
    return app.update_figure_layout(fig, **kwargs)


def use_tonnetz_visualiser(app):
    use_chroma_features(app, ignore_existing=True)
    app.register_visualiser('Tonnetz',
                            ['chroma-features'],
                            tonnetz_visualiser)


def spectral_dome_visualiser(*, features, position, app=WebApp, **kwargs):
    features = app.check_features(features, n=2, asfarray=False)
    fourier_features = np.array(features[0])[:, :, 5]
    chroma_features = np.array(features[1])
    colors = key_colors(chroma_features)
    x, y, z, theta, r = remap_to_xyz(amplitude=fourier_features[0], phase=fourier_features[1], theta_r=True)
    fig = plot_all(x=x, y=y, z=z, colors=colors)
    time_trace = plot_time_traces(x=x, y=y, z=z, colors=colors, times=np.array([position]))[0]
    fig.add_trace(time_trace)
    add_key_markers(
        fig,
        r=r.max(),
        parallels_kwargs=dict(opacity=0.5),
        meridians_kwargs=dict(opacity=0.5),
        label_kwargs=()
    )
    return app.update_figure_layout(fig, **kwargs)


def use_spectral_dome_visualiser(app):
    use_fourier_scape_features(app, ignore_existing=True)
    app.register_visualiser('Spectral Dome',
                            ['fourier-scape-features',
                             'chroma-scape-features'],
                            spectral_dome_visualiser)


def keyscape_visualiser(*, features, position, app=WebApp, dark=False, legend=True, marker_legend=False, **kwargs):
    # lighting for surfaces
    lighting = dict(ambient=1., diffuse=1., roughness=1., specular=1., fresnel=1.)
    # process features
    features = app.check_features(features, n=2, asfarray=False)
    fourier_features = np.array(features[0])[:, :, 5]
    chroma_features = np.array(features[1])
    colors = key_colors(chroma_features)
    # plot keyscape
    x, y, z = remap_to_xyz(amplitude=fourier_features[0], phase=fourier_features[1], scape2D=True)
    fig = create_fig(dark=dark)
    fig = plot_all(x=x, y=y, z=z, colors=colors, fig=fig, do_plot_border=False, do_plot_points=False,
                   plot_surface_kwargs=dict(opacity=1., lighting=lighting))
    fig.add_trace(plot_time_traces(x=x, y=y, z=z,
                                   # colors=colors,
                                   colors=np.ones(x.shape + (3,)) if dark else np.zeros(x.shape + (3,)),
                                   times=np.array([position]))[0])
    # add legend
    if legend:
        # colours
        legend_colors = key_colors(major_minor_profiles[:, (np.arange(12) * 7) % 12, :])  # (mode, trans, RGB)
        major_colours = legend_colors[0]
        minor_colours = legend_colors[1]
        # labels
        lof_shift = -5
        pc_labels = np.roll([str(SpelledPitchClass(value=value)) for value in range(lof_shift, 12 + lof_shift)], lof_shift)
        pc_labels = [l.replace('#', '♯').replace('b', '♭') for l in pc_labels]
        major_labels = [f"{pc}ma" for pc in pc_labels]
        minor_labels = [f"{pc}mi" for pc in pc_labels]
        # rotate to align C major and A minor (etc)
        minor_colours = np.roll(minor_colours, shift=-3, axis=0)
        minor_labels = np.roll(minor_labels, shift=-3)
        # plot
        x = np.repeat([0.88, 0.935], 12).flatten()
        y = np.repeat(np.linspace(0.35, 0.98, 12)[None, :], 2, 0).flatten()
        z = np.zeros(24)
        legend_colors = np.concatenate([major_colours, minor_colours])
        if marker_legend:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                marker_color=legend_colors,
                marker_size=12,
                mode="markers",
                hoverinfo='skip',
                showlegend=False,
            ))
        else:
            for xx, yy, zz, cc in zip(x, y, z, legend_colors):
                fig.add_trace(ellipse_3d([0.025, 0, 0], [0, 0.025, 0], centre=[xx, yy, zz - 0.001],
                                         vertexcolor=[cc] * 101, n=100, lighting=lighting))
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            textposition='middle center',
            texttemplate="<b>%{text}</b>",
            text=np.concatenate([major_labels, minor_labels]),
            mode="text",
            hoverinfo='skip',
            showlegend=False,
        ))
        # to define plot range
        fig.add_trace(go.Scatter3d(x=[0, 1], y=[0, 1], z=[-0.5, 0.5], mode='markers', marker_color=[[0,0,0,0], [0,0,0,0]], showlegend=False))
    return app.update_figure_layout(fig,
                                    **{**dict(
                                        # scene_camera_projection_type='orthographic',
                                        # scene_camera_center=dict(x=0, y=0, z=0),
                                        scene_camera_up=dict(x=0, y=1, z=0),
                                        scene_camera_eye=dict(x=0, y=0, z=1.2),
                                    ),
                                       **kwargs})


def use_keyscape_visualiser(app):
    use_chroma_scape_features(app, ignore_existing=True)
    use_fourier_scape_features(app, ignore_existing=True)
    app.register_visualiser('Keyscape',
                            ['fourier-scape-features',
                             'chroma-scape-features'],
                            keyscape_visualiser)
