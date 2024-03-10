import os
import base64
from io import BytesIO
from itertools import chain
from warnings import warn
from tempfile import NamedTemporaryFile
import logging

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import librosa

from dash import Dash, dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State

from pitchscapes.keyfinding import KeyEstimator
from pitchtypes import EnharmonicPitchClass, SpelledPitchClass

from musicflower.loader import get_chroma, audio_scape
from musicflower.util import get_fourier_component, transpose_profiles, rad_to_deg, tonal_modulo, remap_to_xyz
from musicflower.plotting import rgba, rgba_lighter, rgba_mix, plot_all, key_colors


class WebApp:

    def __init__(self, verbose=False, default_figure_hight=500):
        self.app = None
        self.verbose = verbose
        self.feature_extractors = {}
        self.feature_remappers = {}
        self.visualisers = {}
        self.default_figure_hight = default_figure_hight

    def init(self,
             title="MusicFlower",
             update_title=None,
             sync_interval_ms=50,
             idle_interval_ms=1000,
             n_sync_before_idle=10,
             audio_file=None,
             external_stylesheets=('https://codepen.io/chriddyp/pen/bWLwgP.css',),
             name=None,
             suppress_flask_logger=False,
             ):
        if suppress_flask_logger:
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.app = self._setup_layout(title=title,
                                      idle_interval_ms=idle_interval_ms,
                                      update_title=update_title,
                                      audio_file=audio_file,
                                      external_stylesheets=list(external_stylesheets),
                                      name=name)
        self._setup_audio_position_sync(sync_interval_ms=sync_interval_ms,
                                        idle_interval_ms=idle_interval_ms,
                                        n_sync_before_idle=n_sync_before_idle)
        self._init_download_callbacks()
        self._init_feature_extractor_callbacks()
        self._init_feature_remapper_callbacks()
        self._init_visualiser_callbacks()
        return self

    @classmethod
    def check_features(cls, features, n=1, asfarray=True):
        if len(features) != n:
            raise ValueError(f"expected {n} features but got {len(features)}")
        if n == 1:
            features = features[0]
        if asfarray:
            return np.asfarray(features)
        else:
            return features

    def update_figure_layout(self, figure, **kwargs):
        kwargs = dict(
            height=self.default_figure_hight,
            transition_duration=100,
            margin=dict(t=0, b=10, l=0, r=0),
            uirevision=True,
        ) | kwargs
        figure.update_layout(**kwargs)
        return figure

    def use_chroma_features(self, n=None, name='chroma-features'):
        if n is None:
            self.register_feature_extractor(name, chroma_features)
        else:
            def chroma_features_n(*, audio, app, n=n):
                f = chroma_features(audio=audio, app=app, normalised=False)
                f = downsampler(features=[f], app=app, n=n)
                f = normaliser(features=[f], app=app)
                return f
            self.register_feature_extractor(name, chroma_features_n)
        return self

    def use_chroma_scape_features(self, name='chroma-scape-features', chroma_name=None):
        if chroma_name is None:
            chroma_name = 'chroma-features'
            if chroma_name not in self.feature_extractors and chroma_name not in self.feature_remappers:
                self.use_chroma_features(name=chroma_name)
        self.register_feature_remapper(name, [chroma_name], chroma_scape_features)
        return self

    def use_fourier_features(self, name='fourier-features', chroma_name=None):
        if chroma_name is None:
            chroma_name = 'chroma-features'
            if chroma_name not in self.feature_extractors and chroma_name not in self.feature_remappers:
                self.use_chroma_features(name=chroma_name)
        self.register_feature_remapper(name, [chroma_name], fourier_features)
        return self

    def use_fourier_scape_features(self, name='fourier-scape-features', chroma_name=None):
        if chroma_name is None:
            chroma_name = 'chroma-scape-features'
            if chroma_name not in self.feature_extractors and chroma_name not in self.feature_remappers:
                self.use_chroma_features(name=chroma_name)
        self.register_feature_remapper(name, [chroma_name], fourier_features)
        return self

    def register_feature_extractor(self, name, func):
        """
        Register a feature extractor with given name that computes features from an audio file. The returned features
        are converted to json and stored on the client side. They can be reused by feature remappers.

        :param name: unique name of this feature extractor
        :param func: a callable that takes one 'audio' key-word argument (containing the audio content as a BytesIO
         object, i.e., as if the audio file had been opened using `open` ) and returns the extracted features in a
         format that can be converted to json (e.g. nested lists)
        """
        self._register(key=name, value=func, registry='feature_extractors', msg_name="feature extractor")
        return self

    def register_feature_remapper(self, remapper_name, feature_names, func):
        """
        Register a feature remapper with given name that remaps the input features to a new set of features. Input
        features are retrieved from the client side in json format, so only basic data types are available (e.g.
        arrays are stored as lists).

        :param remapper_name: unique name of this feature remapper
        :param feature_names: iterable with names of the input features to be used
        :param func: a callable that takes one 'features' key-word argument (containing the input features) and
         returns the new feature in a format that can be converted to json (e.g. nested lists)
        """
        if isinstance(feature_names, str):
            raise TypeError(f"'feature_names' is a string ({feature_names}), but should be an iterable of strings")
        feature_names = tuple(feature_names)
        self._register(key=remapper_name, value=(feature_names, func),
                       registry='feature_remappers', msg_name="feature remapper")
        for fname in feature_names:
            if fname not in self.feature_extractors and fname not in self.feature_remappers:
                warn(f"feature remapper '{remapper_name}' requires '{fname}', but no feature extractor or "
                      f"remapper with this name is registered so far", RuntimeWarning)
            return self

    def register_visualiser(self, visualiser_name, feature_names, func, update=False):
        """
        Register a visualiser with given name that generates plots from the input features. Feature values are
        retrieved from the client side in json format, so only basic data types are available (e.g. arrays are stored
        as lists).

        :param visualiser_name: unique name of this visualiser
        :param feature_names: iterable with names of the input features to be used
        :param func: a callable that takes two key-word arguments, 'features' and 'position', (containing the input
         features and the current normalised audio position in the interval [0, 1]) and returns a figure (or dict,
         depending on 'update')
        :param update: this argument can be used to avoid expensive computations that only need to be done once when
         the figure is set up. If 'update' is False (default), 'func' is expected to always return a complete Plotly
         figure at each call; if 'update' is True, 'func' should behave differently depending on whether the
         'position' argument is None or a valid position: If 'position' is None, 'func' should return an entire Plotly
         figure, which is used to initialise the graph; if 'position' is not None (i.e. it is a valid position in [0,
         1]), 'func' should return a list of dictionaries containing updates for the single traces of the existing
         figure, in particular, the 'data' entry of the existing figure is a list of dictionaries defining separate
         traces and each trace will be updated with the corresponding data returned by 'func' (using `new_trace = {
         **old_trace, **update}` so if a key exists in `old_trace` but not in `update`, the old values are retained).
         A typical use case would be to dynamically update the x/y location of a marker while keeping the background
         static. Importantly, in order to update a trace, it has to be present in the initial figure (possibly using
         empty coordinates, to avoid plotting anything). The number of returned dicts must be the same as in the
         number of traces in the initial figure (empty dicts should be used for traces that remain unchanged); also
         the layout is determined by the initial figure as only the trace data is modified.
        """
        if isinstance(feature_names, str):
            raise TypeError(f"'feature_names' is a string ({feature_names}), but should be an iterable of strings")
        feature_names = tuple(feature_names)
        self._register(key=visualiser_name, value=(feature_names, func, update),
                       registry="visualisers", msg_name="visualiser")
        for fname in feature_names:
            if fname not in self.feature_extractors and fname not in self.feature_remappers:
                warn(f"visualiser '{visualiser_name}' requires '{fname}', but no feature extractor or "
                     f"remapper with this name is registered so far", RuntimeWarning)
        return self

    def _register(self, key, value, registry, msg_name):
        self._duplicate_name_check(key)
        getattr(self, registry)[key] = value
        if self.verbose:
            print(f"registered '{key}' {msg_name}")

    def _duplicate_name_check(self, name):
        for reg, reg_name in [(self.feature_extractors, "feature extractor"),
                              (self.feature_remappers, "feature remapper"),
                              (self.visualisers, "visualiser")]:
            if name in reg:
                raise RuntimeError(f"a {reg_name} with name '{name}' already exists")

    def _init_feature_extractor_callbacks(self):
        for name, extractor in self.feature_extractors.items():
            @self.app.callback(
                Output(name, 'data'),
                Input('_audio-content', 'data'),
                prevent_initial_call=True
            )
            def extract_features(audio_content,
                                 name=name, extractor=extractor, app=self):  # bind external variables
                if audio_content is None:
                    if app.verbose:
                        print(f"blocking call of '{name}' feature extractor (audio not available)")
                    return no_update
                else:
                    if app.verbose:
                        print(f"calling '{name}' feature extractor")
                    return extractor(audio=app._file_like_from_upload_content(audio_content), app=app)
            if self.verbose:
                print(f"initialised '{name}' feature extractor")

    def _init_feature_remapper_callbacks(self):
        for name, (feature_names, mapper) in self.feature_remappers.items():
            fname_args = [Input(fname, 'data') for fname in feature_names]
            @self.app.callback(
                Output(name, 'data'),
                *fname_args,
                prevent_initial_call=True
            )
            def feature_remapper(*features, name=name, mapper=mapper, app=self):  # bind external variables
                for f in features:
                    if f is None:
                        if app.verbose:
                            print(f"blocking call of '{name}' feature remapper (input features not available)")
                        return no_update
                if app.verbose:
                    print(f"calling '{name}' feature remapper")
                return mapper(features=features, app=app)
            if self.verbose:
                print(f"initialised '{name}' feature remapper with callbacks for " + ", ".join([f"'{f}'" for f in feature_names]))

    def _init_visualiser_callbacks(self):
        for name, (feature_names, visualiser, update) in self.visualisers.items():

            # A trigger callback that flips the start toggle to indicate the visualiser should run, however, if start
            # and done toggle have different values this means the visualiser is already running and should not be
            # called.
            fname_args = [Input(fname, 'data') for fname in feature_names]
            @self.app.callback(
                Output(component_id=f"_start_toggle_{name}", component_property="disabled"),
                Input(component_id='_stored-audio-position', component_property='value'),
                State(component_id=f"_start_toggle_{name}", component_property="disabled"),
                State(component_id=f"_done_toggle_{name}", component_property="disabled"),
                *fname_args,
                prevent_initial_call=True,
            )
            def start_toggle_func(pos, start, done, *data):
                if done == start:
                    return not start
                else:
                    return no_update

            # The visualiser callback that runs if the start toggle changes value; it sets the done toggle to the same
            # value as the start toggle to indicate it has finished.
            fname_args = [State(fname, 'data') for fname in feature_names]
            @self.app.callback(
                Output(component_id=name, component_property='figure'),
                Output(component_id=f"_done_toggle_{name}", component_property="disabled"),
                Input(component_id=f"_start_toggle_{name}", component_property="disabled"),
                State(component_id='_stored-audio-position', component_property='value'),
                State(component_id=name, component_property='figure'),
                *fname_args,
                prevent_initial_call=True,
            )
            def visualiser_func(start, position, figure, *features,
                                name=name, visualiser=visualiser, update=update, app=self):  # bind external variables
                for f in features:
                    if f is None:
                        if self.verbose:
                            print(f"blocking call of '{name}' visualiser (input features not available)")
                        return no_update, start
                if self.verbose:
                    print(f"calling '{name}' visualiser")
                if update:
                    if not figure or feature_names in ctx.triggered_prop_ids.values():
                        # if data changed: pass None as position to get initial figure
                        return visualiser(features=features, position=None, app=app), start
                    else:
                        # if only position changed: update figure
                        data_updates = visualiser(features=features, position=position, app=app)
                        for d in figure['data']:
                            print(list(d.keys()))
                            for k in d.keys():
                                if k not in ['x', 'y']:
                                    print(f"{k}: {d[k]}")
                        # print(figure['layout'])
                        old_data = figure['data']
                        for idx, new_data_item in enumerate(data_updates):
                            old_data[idx] = {**old_data[idx], **new_data_item}
                        return figure, start
                else:
                    # always get entire figure
                    return visualiser(features=features, position=position, app=app), start
            if self.verbose:
                print(f"initialised '{name}' visualiser with callbacks for " + ", ".join([f"'{f}'" for f in feature_names]))

    def _init_download_callbacks(self):

        # callback to download figure
        @self.app.callback(Output("_download", "data"),
                           Input("_download_figure_btn", "n_clicks"),
                           State('_tab_container', 'value'),
                           State('_tab_container', 'children'),
                           # State(component_id=name, component_property='figure'),
                           prevent_initial_call=True)
        def func(n_nlicks, active_tab, tabs):
            assert active_tab.startswith("_tab_"), active_tab
            filename = f"{active_tab[len('_tab_'):]}.png"
            for tab in tabs:
                if tab['props']['value'] == active_tab:
                    f = go.Figure(tab['props']['children'][0]['props']['figure'])
                    return dcc.send_bytes(f.write_image, filename)
            else:
                raise ValueError(f"Active tab '{active_tab}' not in list of tabs {[t['props']['value'] for t in tabs]}")

    @classmethod
    def _file_like_from_upload_content(cls, content):
        content_type, content_string = content.split(',')
        return BytesIO(base64.b64decode(content_string))

    def _setup_layout(self, title, update_title, idle_interval_ms, audio_file, external_stylesheets, name):
        if name is None:
            name = __name__

        app = Dash(
            name=name,
            title=title,
            update_title=update_title,
            external_stylesheets=list(external_stylesheets),
        )

        # elements in webapp
        layout_content = [
            # html.Div(id='dummy-div', children="", style={'display': 'none'})
        ]

        # audio file upload area and download buttons
        layout_content += [
            # div around everything
            html.Div(children=[
                # div holding the upload area (required for flex display to work)
                html.Div(dcc.Upload(id='_file-upload',
                                    children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                    style={
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False,
                                    ),
                         style={'margin': '5px', 'flex': '1'}),
                # download figure button
                html.Button("Download Figure",
                            id="_download_figure_btn",
                            style={'margin': '5px', 'flex': '0', }),
                # download component
                dcc.Download(id="_download"),
            ],
                style={'display': 'flex'},
            )]

        # callback to display uploaded sound file
        @app.callback(Output('_sound-file-display', 'children'),
                      Output('_audio-content', 'data'),
                      Input('_file-upload', 'contents'),
                      State('_file-upload', 'filename'),
                      Input('_initial-audio-content-update', 'n_intervals'),
                      State('_audio-content', 'data'),
                      prevent_initial_call=True)
        def update_output(contents, name, interval, pre_loaded_contents, verbose=self.verbose):
            if '_initial-audio-content-update' in ctx.triggered_prop_ids.values():
                # initial trigger for preloaded audio file
                if verbose:
                    print(f"updating preloaded audio file")
                return no_update, pre_loaded_contents
            if contents is not None:
                # new audio file was uploaded
                return html.Div([
                    html.P(name),
                    html.Audio(src=contents, controls=True, id='_audio-controls', style={'width': '100%'}),
                ]), contents
            else:
                return no_update, no_update

        # tabs for all registered visualisers
        layout_content += [dcc.Tabs(
            id="_tab_container",
            value=[f"_tab_{name}" for name in self.visualisers][0],  # make first tab active
            children=[dcc.Tab(label=name,
                              id=f"_tab_{name}",
                              value=f"_tab_{name}",
                              children=[dcc.Graph(figure={}, id=name)]) for name in self.visualisers])]

        # toggles to trigger/indicate running of visualisers
        layout_content += [
            html.Div(children=[html.Button(
                id=f"_start_toggle_{name}",
                children=f"{name} (start)",
                style={'display': 'none'}) for name in self.visualisers]),
            html.Div(children=[html.Button(
                id=f"_done_toggle_{name}",
                children=f"{name} (done)",
                style={'display': 'none'}
            ) for name in self.visualisers]),]

        # if audio file was provided, load it; else just provide div for later displaying uploaded file
        if audio_file is not None:
            encoded_sound = base64.b64encode(open(audio_file, 'rb').read())
            decoded_sound = encoded_sound.decode()
            extension = audio_file.split('.')[-1].lower()
            try:
                extension = {'mp3': 'mpeg'}[extension]
            except KeyError:
                pass
            audio_src = f'data:audio/{extension};base64,{decoded_sound}'
            layout_content += [
                html.Div(id='_sound-file-display', children=html.Div([
                    html.P(audio_file),
                    html.Audio(src=audio_src, controls=True, id='_audio-controls', style={'width': '100%'}),
                ])),
                dcc.Store(id='_audio-content', data=audio_src),
                dcc.Interval(id='_initial-audio-content-update', interval=5000, max_intervals=1)
            ]
        else:
            layout_content += [html.Div(id='_sound-file-display'),
                               dcc.Store(id='_audio-content', data=None),
                               dcc.Interval(id='_initial-audio-content-update', disabled=True)]

        # stores for results of feature extractors and feature remappers
        for feature_name in chain(self.feature_extractors, self.feature_remappers):
            layout_content += [dcc.Store(id=feature_name, data=None)]

        # some invisible elements
        input_kwargs = dict(style={'display': 'none'})
        # input_kwargs = {}
        layout_content += [
            # timer to poll/sync playback position from audio element
            dcc.Interval(id='_audio-sync-timer', interval=idle_interval_ms),
            # current audio position to be synced with audio element
            dcc.Input(id='_current-audio-position', value='', **input_kwargs),
            # stored audio position to detect changes
            dcc.Input(id='_stored-audio-position', value='', **input_kwargs),
            # remember idle polls to adapt poll frequency
            dcc.Input(id='_n-const-polls', value=0, type='number', **input_kwargs),
        ]

        app.layout = html.Div(layout_content)

        return app

    def _setup_audio_position_sync(self, sync_interval_ms, idle_interval_ms, n_sync_before_idle):

        # poll and store audio position
        self.app.clientside_callback(
            '''
            function(value) {
                const audio = document.getElementById('_audio-controls');
                if (audio == null) {
                    return null
                } else {
                    return audio.currentTime / audio.duration;
                }
            }
            ''',
            Output(component_id='_current-audio-position', component_property='value'),
            Input(component_id='_audio-sync-timer', component_property='n_intervals'),
            prevent_initial_call=True
        )

        # update stored audio position if changed; adapt polling interval
        @self.app.callback(
            Output(component_id='_stored-audio-position', component_property='value'),
            Output('_audio-sync-timer', 'interval'),
            Output('_n-const-polls', 'value'),
            Input(component_id='_current-audio-position', component_property='value'),
            State(component_id='_stored-audio-position', component_property='value'),
            State('_n-const-polls', 'value'),
            State('_audio-sync-timer', 'interval'),
            prevent_initial_call=True
        )
        def update_audio_position(current_pos, stored_pos, n_const_polls, current_interval):
            # check if current position is undefined (None) or has not changed
            if current_pos is None or current_pos == stored_pos:
                # if number of idle polls is above threshold and interval is not idle yet: change
                # otherwise just increment idle counts
                if n_const_polls > n_sync_before_idle and current_interval != idle_interval_ms:
                    # return no_update, no_update, idle_interval_ms, n_const_polls + 1
                    return no_update, idle_interval_ms, n_const_polls + 1
                else:
                    # return no_update, no_update, no_update, n_const_polls + 1
                    return no_update, no_update, n_const_polls + 1
            else:
                # if position has changed: update stored position
                # also set sync interval if necessary, and reset idle counter
                if current_interval != sync_interval_ms:
                    # return frame, current_pos, sync_interval_ms, 0
                    return current_pos, sync_interval_ms, 0
                else:
                    # return frame, current_pos, no_update, 0
                    return current_pos, no_update, 0

    def run(self, *args, **kwargs):
        self.app.run_server(*args, **kwargs)


def none_feature(*, audio, app):
    return []


def waveform_feature(*, audio, app, use_real_file=True):
    if use_real_file:
        # create temporary file as librosa.load works better with a real file
        tf = NamedTemporaryFile(delete=False)
        # write audio content to temporary file
        tf.write(audio.read())
        # load audio from file
        y, sr = librosa.load(tf.name)
        # close and remove temporary file
        tf.close()
        os.unlink(tf.name)
    else:
        y, sr = librosa.load(audio)
    return y, sr


def spectrogram_features(*, audio, app):
    y, sr = waveform_feature(audio=audio, app=app, use_real_file=True)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db.T


def chroma_features(*, audio, app, normalised=True):
    y, sr = waveform_feature(audio=audio, app=app, use_real_file=True)
    chroma = get_chroma(
        data=(y, sr),
        # hop_length=512,  # default
        # hop_length=2048,
    ).T
    if normalised:
        chroma = normaliser(features=chroma, app=app)
    return chroma


def chroma_scape_features(*, features, app):
    features = app.check_features(features)
    return audio_scape(n_time_intervals=features.shape[0], raw_chroma=features.T, top_down=True)


def normaliser(*, features, app, inplace=True, axis=-1):
    features = app.check_features(features)
    if inplace:
        features /= features.sum(axis=axis, keepdims=True)
        return features
    else:
        return features / features.sum(axis=axis, keepdims=True)


def fourier_features(*, features, app):
    """
    Compute the amplitude and phase of all Fourier components using `get_fourier_component`.
    """
    features = app.check_features(features)
    return get_fourier_component(features)


def downsampler(*, features, app, n):
    # downsample by computing mean over window
    features = app.check_features(features)
    length = features.shape[0]
    if length < n:
        warn(f"Cannot downsample: requested size ({n}) is larger than input size ({length}), returning input")
        return features
    batch_shape = features.shape[1:]
    downsampled = np.zeros((n,) + batch_shape)
    for idx in range(n):
        downsampled[idx, ...] = features[int(length * idx / n):int(length * (idx + 1) / n), ...].mean(axis=0)
    return downsampled


def get_downsampler(app, n):
    return lambda features: downsampler(features=features, app=app, n=n)


def waveform_visualiser(*, features, position, app, update=True):
    if update:
        if position is None:
            y, sr = features
            y = np.array(y)
            waveform = px.line(y)
            return go.Figure(data=[
                waveform.data[0],
                go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, dash="dash", color="red"))
            ], layout=waveform.layout)
        else:
            x_max = (len(features[0]) - 1)
            pos = position * x_max
            return [{}, dict(x=[pos, pos], y=[-1, 1])]
    else:
        y, sr = features
        y = np.array(y)
        fig = px.line(y)
        fig.add_vline(position * (y.shape[0] - 1), line_width=3, line_dash="dash", line_color="red", opacity=0.9)
        return fig


def heatmap_visualiser(*, features, position, app, update=False, express=True):
    features = app.check_features(features)
    if update:
        if position is None:
            heatmap = px.imshow(features.T, origin='lower', aspect="auto")
            return go.Figure(data=[
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
            return fig
        else:
            heatmap = px.imshow(features.T, origin='lower', aspect="auto")
            return go.Figure(data=[
                heatmap.data[0],
                go.Scatter(x=[pos, pos], y=[0, y_max], mode='lines', line=dict(width=3, dash="dash", color="red"))
            ], layout=heatmap.layout)


def advanced_chroma_visualiser_fast(*, features, position, app):
    features = app.check_features(features)
    x_max = (features.shape[0] - 1)
    if position is None:
        heatmap = px.imshow(features.T, origin='lower', aspect="auto")
        return go.Figure(data=[
            heatmap.data[0],
            go.Scatter(x=[], y=[], mode='lines', line=dict(width=3, dash="dash", color="red"))
        ], layout=heatmap.layout)
    else:
        pos = position * x_max
        return [{}, dict(x=[pos, pos], y=[0, 11])]


def single_fourier(*, features, position, app, component):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    fig = px.line_polar(r=features[0, :, component], theta=features[1, :, component])
    idx = int(position * (features.shape[1] - 1))
    fig.add_trace(go.Scatterpolar(
        r=features[0, idx:idx + 1, component],
        theta=features[1, idx:idx + 1, component],
        mode='markers',
        marker=dict(size=10, color='red')
    ))
    return fig


def fourier_visualiser(*, features, position, app, binary_profiles=True, incl=None):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    labels = [f"{nth} Coefficient" for nth in ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th']]
    specs = [[{'type': 'polar'} for _ in range(3)] for _ in range(2)]
    specs[1][2] = {}
    fig = make_subplots(rows=2, cols=3, start_cell="top-left", specs=specs,
                        # subplot_titles=labels
                        )
    idx = int(position * (features.shape[1] - 1))
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
        if component == 5:
            if binary_profiles:
                # key signatures represented by binary profiles
                profiles = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7
                profiles = transpose_profiles(profiles)  # (trans, pitch)
                fourier_profiles = fourier_features(profiles)  # (mag/phase, trans, component)
                fig.add_trace(go.Scatterpolar(
                    hoverinfo='skip',
                    r=fourier_profiles[0, :, component],
                    theta=fourier_profiles[1, :, component] * rad_to_deg,
                    mode='markers',
                    marker=dict(size=2, color='black'),
                    showlegend=False,
                ), row=row, col=col)
                # set ticks
                angularaxis = dict(
                    tickmode='array',
                    tickvals=np.mod(fourier_profiles[1, :, component] * rad_to_deg, 360),
                    ticktext=tonal_modulo(np.array(
                        ["♮", "♯", "2♯", "3♯", "4♯", "5♯", "6♯", "5♭", "4♭", "3♭", "2♭", "♭"]
                    )),
                )
            else:
                # major/minor keys represented by Albrecht profiles
                profiles = KeyEstimator.profiles['albrecht']
                profiles = np.array([profiles['major'], profiles['minor']]).T  # (pitch, mode)
                profiles = transpose_profiles(profiles)  # (trans, pitch, mode)
                profiles = np.moveaxis(profiles, 2, 0)  # (mode, trans, pitch)
                fourier_profiles = fourier_features(profiles)  # (mag/phase, mode, trans, component)
                fig.add_trace(go.Scatterpolar(
                    hoverinfo='skip',
                    r=fourier_profiles[0, :, :, component].flatten(),
                    theta=fourier_profiles[1, :, :, component].flatten() * rad_to_deg,
                    mode='markers',
                    marker=dict(size=2, color='black'),
                    showlegend=False,
                ), row=row, col=col)
                # set ticks
                pitch_names = [str(EnharmonicPitchClass(n)) for n in range(12)]
                ma_mi_phase = np.concatenate([fourier_profiles[1, 0, :, component], fourier_profiles[1, 1, :, component]])
                angularaxis = dict(
                    tickmode='array',
                    tickvals=np.mod(ma_mi_phase * rad_to_deg, 360),
                    ticktext = np.array([[f"{n}ma", f"{n}mi"] for n in pitch_names]).T.flatten()
                )
        else:
            angularaxis = {}
        # adjust axes
        fig.update_polars(
            radialaxis=dict(range=[0, 1], showticklabels=False, ticks=''),
            angularaxis=angularaxis,
            # angularaxis=dict(showticklabels=False, ticks=''),
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
    return app.update_figure_layout(fig)


def circle_of_fifths_visualiser(*, features, position, app, ticks="binary"):
    features = app.check_features(features)
    features[1] *= rad_to_deg
    idx = int(position * (features.shape[1] - 1))
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
    fourier_profiles = fourier_features(profiles)  # (mag/phase, trans, component)
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
    profiles = KeyEstimator.profiles['albrecht']
    profiles = np.array([profiles['major'], profiles['minor']]).T  # (pitch, mode)
    profiles = transpose_profiles(profiles)  # (trans, pitch, mode)
    profiles = np.moveaxis(profiles, 2, 0)  # (mode, trans, pitch)
    fourier_profiles = fourier_features(profiles)  # (mag/phase, mode, trans, component)
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
    return app.update_figure_layout(fig)


def tonnetz_visualiser(*, features, position, app):
    features = app.check_features(features, asfarray=False)
    pos_idx = int(np.round(position * (len(features) - 1)))
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
    return app.update_figure_layout(fig)


def keyscape_3d(*, features, position, app):
    features = app.check_features(features, n=2, asfarray=False)
    fourier_features = np.array(features[0])[:, :, 5]
    chroma_features = np.array(features[1])
    colors = key_colors(chroma_features)
    x, y, z = remap_to_xyz(amplitude=fourier_features[0], phase=fourier_features[1])
    fig = plot_all(x=x, y=y, z=z, colors=colors)
    return app.update_figure_layout(fig)


if __name__ == '__main__':

    app = WebApp(
        verbose=True,
        default_figure_hight=800,
        # default_figure_hight=1000,
    )

    # app.register_feature_extractor('None', none_feature)
    # app.register_feature_extractor('waveform-feature', waveform_feature)
    # app.register_feature_extractor('spectrogram-features', spectrogram_features)
    app.use_chroma_features(100)
    app.use_fourier_features()
    app.use_chroma_scape_features()
    app.use_fourier_scape_features()

    # app.register_visualiser('Waveform', ['waveform-feature'], waveform_visualiser, update=True)
    # app.register_visualiser('Spectrogram', ['spectrogram-features'], heatmap_visualiser)
    app.register_visualiser('Chroma Features', ['chroma-features'], heatmap_visualiser)
    # app.register_visualiser('Chroma 100', ['chroma-100'], heatmap_visualiser, update=False)
    # app.register_visualiser('Fourier Magnitude', ['fourier-mag'], heatmap_visualiser, update=False)
    # app.register_visualiser('Fourier Phase', ['fourier-phase'], heatmap_visualiser, update=False)

    # app.register_visualiser('Fourier Coefficients', ['fourier-features'], fourier_visualiser)
    # app.register_visualiser('Circle of Fifths', ['fourier-features'], circle_of_fifths_visualiser)
    # app.register_visualiser('Single Fourier', ['fourier-features'], lambda **kwargs: single_fourier(component=5, **kwargs))
    app.register_visualiser('Tonnetz', ['chroma-features'], tonnetz_visualiser)
    app.register_visualiser('3D Keyscape', ['fourier-scape-features', 'chroma-scape-features'], keyscape_3d)

    # app.register_visualiser('Express Chroma Heatmap', ['chroma-features'], heatmap_visualiser, update=False)
    # app.register_visualiser('Chroma Heatmap (static)', ['chroma-features'], advanced_chroma_visualiser_slow, update=False)
    # app.register_visualiser('Chroma Heatmap (update)', ['chroma-features'], advanced_chroma_visualiser_fast, update=True)

    app.init(
        title="<library>",
        # update_title='Updating...',
        suppress_flask_logger=True,  # suppress extensive logging, only show errors
        audio_file='../Shepard_Tone.wav',
        # audio_file='../J.S. Bach - Prelude in C Major.mp3'
        # sync_interval_ms=300,
        external_stylesheets=[],
    )

    # @app.app.callback(Output('dummy-div', 'children'),
    #                   Input('chroma-features', 'data'),
    #                   Input('_stored-audio-position', 'value'),
    #                   prevent_initial_call=True)
    # def test(data, value):
    #     print(f"FIRED! {type(data)}/{value} | {ctx.triggered_id} | {ctx.triggered_prop_ids}")

    app.run(
        # debug=False,
        debug=True,
    )