import os
import base64
from io import BytesIO
from itertools import chain
from warnings import warn
from tempfile import NamedTemporaryFile, TemporaryDirectory
import logging
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import librosa

from dash import Dash, dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from pitchscapes.keyfinding import KeyEstimator
from pitchtypes import EnharmonicPitchClass, SpelledPitchClass

from musicflower.loader import get_chroma, audio_scape
from musicflower.util import get_fourier_component, transpose_profiles, rad_to_deg, tonal_modulo, remap_to_xyz
from musicflower.plotting import (rgba, rgba_lighter, rgba_mix, plot_all, key_colors, add_key_markers, plot_time_traces,
                                  create_fig, ellipse_3d)


class WebApp:

    major_minor_profiles = KeyEstimator.profiles['albrecht']
    major_minor_profiles = np.array([major_minor_profiles['major'], major_minor_profiles['minor']]).T  # (pitch, mode)
    major_minor_profiles = transpose_profiles(major_minor_profiles)  # (trans, pitch, mode)
    major_minor_profiles = np.moveaxis(major_minor_profiles, 2, 0)  # (mode, trans, pitch)

    def __init__(self, verbose=False):
        self.app = None
        self.verbose = verbose
        self.feature_extractors = {}
        self.feature_remappers = {}
        self.visualisers = {}
        self._visualiser_callbacks = {}
        self.visualiser_default_inputs = (
            ('_figure_height', 'value', 'height'),
            ('_figure_width', 'value', 'width'),
        )
        self.visualiser_default_input_states = [State(elem_id, value)
                                                for elem_id, value, _ in self.visualiser_default_inputs]
        self.visualiser_default_input_names = tuple(name
                                                    for _, _, name in self.visualiser_default_inputs)

    def init(self,
             title="MusicFlower",
             update_title=None,
             sync_interval_ms=50,
             idle_interval_ms=1000,
             n_sync_before_idle=10,
             audio_file=None,
             external_stylesheets=(
                     # 'https://codepen.io/chriddyp/pen/bWLwgP.css',
             ),
             name=None,
             suppress_flask_logger=True,
             figure_width=None,
             figure_height=None,
             _debug_display_toggles=False,
             dash_kwargs=(),
             ):
        """

        :param title:
        :param update_title:
        :param sync_interval_ms:
        :param idle_interval_ms:
        :param n_sync_before_idle:
        :param audio_file:
        :param external_stylesheets:
        :param name:
        :param suppress_flask_logger: suppress extensive logging, only show errors
        :param figure_width:
        :param figure_height:
        :param _debug_display_toggles:
        :param dash_kwargs:
        :return:
        """
        self._debug_display_toggles = _debug_display_toggles
        external_stylesheets = list(external_stylesheets) + [dbc.themes.BOOTSTRAP]  # always need that for settings modal
        if suppress_flask_logger:
            logging.getLogger('werkzeug').setLevel(logging.ERROR)
        self.app = self._setup_layout(title=title,
                                      idle_interval_ms=idle_interval_ms,
                                      update_title=update_title,
                                      audio_file=audio_file,
                                      external_stylesheets=external_stylesheets,
                                      name=name,
                                      figure_width=figure_width,
                                      figure_height=figure_height,
                                      dash_kwargs=dash_kwargs)
        self._setup_audio_position_sync(sync_interval_ms=sync_interval_ms,
                                        idle_interval_ms=idle_interval_ms,
                                        n_sync_before_idle=n_sync_before_idle)
        self._init_feature_extractor_callbacks()
        self._init_feature_remapper_callbacks()
        self._init_visualiser_callbacks()
        return self

    @classmethod
    def check_features(cls, features, n=1, asfarray=True):
        if len(features) != n:
            raise ValueError(f"expected {n} features but got {len(features)}")
        if asfarray:
            features = [np.asarray(f, dtype=float) for f in features]
        if n == 1:
            features = features[0]
        return features

    @classmethod
    def update_figure_layout(cls, figure, **kwargs):
        # remove None values
        for v in ['height', 'width']:
            if v in kwargs and kwargs[v] is None:
                del kwargs[v]
        # populate with some default values
        kwargs = dict(
            transition_duration=10,
            margin=dict(t=0, b=10, l=0, r=0),
            uirevision=True,
        ) | kwargs
        # update and return figure
        figure.update_layout(**kwargs)
        return figure

    @classmethod
    def position_idx(cls, position, *, n=None, features=None):
        """
        For a position in [0, 1] and features of length n, compute the corresponding index in {0, ..., n - 1}.

        :param position: number in [0, 1]
        :param n: length of features
        :param features: features to compute n as len(features)
        :return: index
        """
        if (n is None) == (features is None):
            raise ValueError("Either 'n' or 'features' have to be provided (not both)")
        if features is not None:
            n = len(features)
        return int(round(position * (n - 1)))

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

            # callback function
            def visualiser_callback(features, position, app, figure, name=name, visualiser=visualiser, update=update, **kwargs):
                for f in features:
                    if f is None:
                        if app.verbose:
                            print(f"blocking call of '{name}' visualiser (input features not available)")
                        return None
                if app.verbose:
                    print(f"calling '{name}' visualiser")
                if update:
                    if not figure or feature_names in ctx.triggered_prop_ids.values():
                        # if data changed: pass None as position to get initial figure
                        figure = visualiser(features=features, position=None, app=app, **kwargs)
                    else:
                        # if only position changed: update figure
                        data_updates = visualiser(features=features, position=position, app=app, **kwargs)
                        old_data = figure['data']
                        for idx, new_data_item in enumerate(data_updates):
                            old_data[idx] = {**old_data[idx], **new_data_item}
                else:
                    # always get entire figure
                    figure = visualiser(features=features, position=position, app=app, **kwargs)
                return figure

            # remember for later use
            self._visualiser_callbacks[name] = visualiser_callback

            # feature inputs for this visualiser (as Input, so they trigger rendering)
            feature_inputs = [Input(fname, 'data') for fname in feature_names]

            # feature inputs for this visualiser (as State, rendering is triggered above)
            feature_states = [State(fname, 'data') for fname in feature_names]

            #######################
            #  DYNAMIC RENDERING  #
            #######################

            # A trigger callback that flips the start toggle to indicate the visualiser should run, however, if start
            # and done toggle have different values this means the visualiser is already running and should not be
            # called.
            @self.app.callback(
                Output(component_id=f"_start_toggle_{name}", component_property="disabled"),
                Input(component_id='_stored-audio-position', component_property='value'),
                State(component_id=f"_start_toggle_{name}", component_property="disabled"),
                State(component_id=f"_done_toggle_{name}", component_property="disabled"),
                *feature_inputs,
                prevent_initial_call=True,
            )
            def start_toggle_func(pos, start, done, *data, name=name):
                if done == start:
                    # print(f"activating '{name}'")
                    return not start
                else:
                    # print(f"'{name}' already running")
                    return no_update

            # The visualiser callback that runs if the start toggle changes value; it sets the done toggle to the same
            # value as the start toggle to indicate it has finished.
            @self.app.callback(
                Output(component_id=name, component_property='figure'),
                Output(component_id=f"_done_toggle_{name}", component_property="disabled"),
                Input(component_id=f"_start_toggle_{name}", component_property="disabled"),
                State(component_id='_stored-audio-position', component_property='value'),
                State(component_id=name, component_property='figure'),
                *self.visualiser_default_input_states,
                *feature_states,
                prevent_initial_call=True,
            )
            def visualiser_func(start, position, figure, *args, name=name, app=self):  # bind external variables
                kwargs, features = app._unpack_visualiser_args(*args)
                figure = app._visualiser_callbacks[name](features=features, position=position, app=app, figure=figure, **kwargs)
                if figure is None:
                    return no_update, start
                else:
                    return figure, start

            #################
            #  SAVE FIGURE  #
            #################

            # trigger save figure callback
            @self.app.callback(Output(component_id=f"_save_figure_toggle_{name}", component_property="disabled"),
                               Input("_download_figure_btn", "n_clicks"),
                               State('_tab_container', 'value'),
                               State(component_id=f"_save_figure_toggle_{name}", component_property="disabled"),
                               prevent_initial_call=True)
            def func(n_nlicks, active_tab, toggle_value, name=name):
                assert active_tab.startswith("_tab_"), active_tab
                if active_tab == "_tab_" + name:
                    return not toggle_value  # change toggle to trigger save figure callback
                else:
                    return no_update

            # The visualiser callback to save the figure
            @self.app.callback(
                Output(f"_save_figure_download_{name}", "data"),
                Input(component_id=f"_save_figure_toggle_{name}", component_property="disabled"),
                State(component_id='_stored-audio-position', component_property='value'),
                State(component_id=name, component_property='figure'),
                *self.visualiser_default_input_states,
                *feature_states,
                prevent_initial_call=True,
            )
            def visualiser_func(toggle, position, figure, *args, name=name, app=self):  # bind external variables
                if app.verbose:
                    print(f"saving figure: {name}")
                kwargs, features = app._unpack_visualiser_args(*args)
                figure = app._visualiser_callbacks[name](features=features, position=position, app=app, figure=figure, **kwargs)
                return dcc.send_bytes(
                    # figure.write_image,
                    lambda *a, **k: figure.write_image(*a, **k,
                                                       engine="kaleido",  # (default) marker size incorrect + problems when only ambient lighting is on
                                                       # engine="orca",   # only marker size incorrect
                                                       width=kwargs['width'],
                                                       height=kwargs['height'],
                                                       ),
                    "figure.png")

            ################
            #  SAVE VIDEO  #
            ################

            # trigger save video callback
            @self.app.callback(Output(component_id=f"_save_video_toggle_{name}", component_property="disabled"),
                               Input("_download_video_btn", "n_clicks"),
                               State('_tab_container', 'value'),
                               State(component_id=f"_save_video_toggle_{name}", component_property="disabled"),
                               prevent_initial_call=True)
            def func(n_nlicks, active_tab, toggle_value, name=name):
                assert active_tab.startswith("_tab_"), active_tab
                if active_tab == "_tab_" + name:
                    return not toggle_value  # change toggle to trigger save video callback
                else:
                    return no_update

            # The visualiser callback to save the video
            @self.app.callback(
                Output(f"_save_video_download_{name}", "data"),
                Input(component_id=f"_save_video_toggle_{name}", component_property="disabled"),
                State("_number_of_frames", "value"),
                State(component_id=name, component_property='figure'),
                State('_audio-content', 'data'),
                State(component_id='_audio-duration', component_property='value'),
                State('_audio_filename', 'value'),
                *self.visualiser_default_input_states,
                *feature_states,
                prevent_initial_call=True,
            )
            def visualiser_func(toggle, resolution, figure, audio, duration, audio_filename, *args, name=name, app=self):  # bind external variables
                if app.verbose:
                    print(f"saving video for '{name}'...")
                kwargs, features = app._unpack_visualiser_args(*args)
                # save figures to temporary directory
                framerate = resolution / duration
                with TemporaryDirectory() as tmpdir:
                    if app.verbose:
                        print('created temporary directory', tmpdir)
                        print(f"framerate: {framerate}")
                    # create audio file
                    audio_file_path = Path(tmpdir) / f"{name}_{audio_filename}"
                    audio = app._file_like_from_upload_content(audio)
                    with open(audio_file_path, 'wb') as a:
                        a.write(audio.read())
                    # create frames/figures
                    for pos in np.linspace(0, 1, resolution):
                        figure = app._visualiser_callbacks[name](features=features, position=pos, app=app, figure=figure, **kwargs)
                        filename = Path(tmpdir) / f"{name}_{pos}.png"
                        if app.verbose:
                            print(f"writing frame '{filename}'")
                        figure.write_image(filename)
                    # create video file
                    video_file = Path(tmpdir) / 'video.mp4'
                    # ffmpeg (see e.g. https://trac.ffmpeg.org/wiki/Slideshow)
                    # using "-max_muxing_queue_size 4096": https://stackoverflow.com/questions/49686244/ffmpeg-too-many-packets-buffered-for-output-stream-01
                    ffmpeg_command = f"/bin/ffmpeg -framerate {framerate} -pattern_type glob -i '{Path(tmpdir)/'*.png'}' -i '{audio_file_path}' -r 30 -c:v libx264 -max_muxing_queue_size 4096 -y {video_file}"
                    if app.verbose:
                        files = list(sorted(os.listdir(tmpdir)))
                        print(files)
                        print(f"calling ffmpeg: {ffmpeg_command}")
                    os.system(ffmpeg_command)
                    return dcc.send_file(video_file)

            ################
            #  PRINT INFO  #
            ################
            if self.verbose:
                print(f"initialised '{name}' visualiser with callbacks for " + ", ".join([f"'{f}'" for f in feature_names]))

    def _unpack_visualiser_args(self, *args):
        n_default_inputs = len(self.visualiser_default_input_names)
        kwargs = dict(zip(self.visualiser_default_input_names, args[:n_default_inputs]))
        features = args[n_default_inputs:]
        return kwargs, features

    @classmethod
    def _file_like_from_upload_content(cls, content):
        content_type, content_string = content.split(',')
        return BytesIO(base64.b64decode(content_string))

    def _setup_layout(self, title, update_title, idle_interval_ms, audio_file, external_stylesheets, name,
                      figure_width, figure_height, dash_kwargs=()):
        if name is None:
            name = __name__

        dash_kwargs = {
            **dict(name=name,
                   title=title,
                   update_title=update_title,
                   external_stylesheets=list(external_stylesheets),
                   assets_folder=Path(os.path.dirname(os.path.realpath(__file__))) / "assets"),
            **dict(dash_kwargs)
        }
        app = Dash(**dash_kwargs)

        # elements in webapp
        layout_content = [
            # html.Div(id='dummy-div', children="", style={'display': 'none'})
        ]

        # audio file upload area and download and settings buttons
        head_height = '60px'
        button_style = {'margin': '5px', 'flex': f'0 0 {head_height}', 'height': head_height}
        layout_content += [
            # div around everything
            html.Div(children=[
                # div holding the upload area (required for flex display to work)
                html.Div(dcc.Upload(id='_file-upload',
                                    children=html.Div(['Drag and Drop or Click to Select File']),
                                    style={
                                        'height': head_height,
                                        'lineHeight': head_height,
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False,
                                    ),
                         style={'margin': '5px', 'flex': '1'}),
                # download figure button
                html.Button(html.I(className='fas fa-2x fa-camera'), id="_download_figure_btn", style=button_style),
                # download video button
                html.Button(html.I(className='fas fa-2x fa-video'), id="_download_video_btn", style=button_style),
                # settings button
                html.Button(html.I(className='fas fa-2x fa-cog'), id="_settings_btn", style=button_style),
            ],
                style={'display': 'flex'},
            )]

        # callback to display uploaded sound file
        @app.callback(Output('_sound-file-display', 'children'),
                      Output('_audio-content', 'data'),
                      Output('_audio_filename', 'value'),
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
                return no_update, pre_loaded_contents, no_update
            if contents is not None:
                # new audio file was uploaded
                return self._audio_element(audio_file=name, audio_src=contents), contents, name
            else:
                return no_update, no_update, no_update

        # settings modal
        flex_row_style = {'display': 'flex'}
        flex_col_style = {'flex': 1}
        layout_content += [dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Settings")),
                dbc.ModalBody(children=[
                    html.Div(children=[
                        html.Div(html.Label(html.Strong("Figure width:")), style=flex_col_style),
                        html.Div(dcc.Input(id="_figure_width", type="number", value=figure_width, min=1, max=10000, step=1), style=flex_col_style),
                    ], style=flex_row_style),
                    html.Div(children=[
                        html.Div(html.Label(html.Strong("Figure height:")), style=flex_col_style),
                        html.Div(dcc.Input(id="_figure_height", type="number", value=figure_height, min=1, max=10000, step=1), style=flex_col_style),
                    ], style=flex_row_style),
                    html.Div(children=[
                        html.Div(html.Label(html.Strong("Number of frames (video):")), style=flex_col_style),
                        html.Div(dcc.Input(id="_number_of_frames", type="number", value=10, min=1, max=10000, step=1), style=flex_col_style),
                    ], style=flex_row_style),
                ]),
                dbc.ModalFooter(
                    dbc.Button(
                        "Close", id="_close_settings_modal", className="ms-auto", n_clicks=0
                    )
                ),
            ],
            id="_settings_modal",
            is_open=False,
        )]

        # callback to open settings
        @app.callback(Output("_settings_modal", "is_open"),
                      Input("_settings_btn", "n_clicks"),
                      Input("_close_settings_modal", "n_clicks"),
                      State("_settings_modal", "is_open"),
                      prevent_initial_call=True)
        def toggle_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open

        # tabs for all registered visualisers
        layout_content += [dcc.Tabs(
            id="_tab_container",
            value=[f"_tab_{name}" for name in self.visualisers][0],  # make first tab active
            children=[dcc.Tab(label=name,
                              id=f"_tab_{name}",
                              value=f"_tab_{name}",
                              children=[
                                  html.Div([
                                      html.Div(style={'flex': 1}),
                                      html.Div(dcc.Graph(figure={}, id=name), style={'flex': 1}),
                                      html.Div(style={'flex': 1}),
                                  ], style={'display': 'flex'},)
                              ]) for name in self.visualisers])]

        # toggles/triggers for visualisers/tabs:
        # trigger running, indicate running, trigger save figure, trigger save video
        layout_content += [
            html.Div(children=[
                html.Button(id=f"_{role}_toggle_{name}", children=f"{name} ({role})",
                            style=({} if self._debug_display_toggles else {'display': 'none'}))
                for role in ["start", "done", "save_figure", "save_video"]
                for name in self.visualisers])]

        # download components for visualisers
        layout_content += [
            html.Div(children=[
                dcc.Download(id=f"_save_{role}_download_{name}")
                for name in self.visualisers
                for role in ["figure", "video"]])
        ]

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
                self._audio_element(audio_file=audio_file, audio_src=audio_src),
                dcc.Store(id='_audio-content', data=audio_src),
                dcc.Interval(id='_initial-audio-content-update', interval=5000, max_intervals=1)
            ]
        else:
            layout_content += [
                self._audio_element(audio_file=audio_file),
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
            # audio duration
            dcc.Input(id='_audio-duration', value='', **input_kwargs),
            # current audio position to be synced with audio element
            dcc.Input(id='_current-audio-position', value='', **input_kwargs),
            # stored audio position to detect changes
            dcc.Input(id='_stored-audio-position', value='', **input_kwargs),
            # remember idle polls to adapt poll frequency
            dcc.Input(id='_n-const-polls', value=0, type='number', **input_kwargs),
            #
            dcc.Input(id='_audio_filename', value='audio', **input_kwargs),
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
                    return null;
                } else {
                    return [audio.currentTime / audio.duration, audio.duration];
                }
            }
            ''',
            Output(component_id='_current-audio-position', component_property='value'),
            Output(component_id='_audio-duration', component_property='value'),
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

    def _audio_element(self, audio_file, audio_src=None):
        src = {} if audio_src is None else dict(src=audio_src)
        return html.Div(id='_sound-file-display', children=html.Div([
            html.P(audio_file),
            html.Audio(**src, controls=True, id='_audio-controls', style={'width': '100%'}),
        ]))

    def run(self, *args, **kwargs):
        self.app.run(*args, **kwargs)


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
        chroma = normaliser(features=[chroma], app=app)
    return chroma


def chroma_scape_features(*, features, app=None):
    features = WebApp.check_features(features)
    return audio_scape(n_time_intervals=features.shape[0], raw_chroma=features.T, top_down=True)


def normaliser(*, features, app=None, inplace=True, axis=-1):
    features = WebApp.check_features(features)
    if inplace:
        features /= features.sum(axis=axis, keepdims=True)
        return features
    else:
        return features / features.sum(axis=axis, keepdims=True)


def fourier_features(*, features, app=None):
    """
    Compute the amplitude and phase of all Fourier components using `get_fourier_component`.
    """
    features = WebApp.check_features(features)
    return get_fourier_component(features)


def downsampler(*, features, app=None, n):
    # downsample by computing mean over window
    features = WebApp.check_features(features)
    length = features.shape[0]
    if length < n:
        warn(f"Cannot downsample: requested size ({n}) is larger than input size ({length}), returning input")
        return features
    batch_shape = features.shape[1:]
    downsampled = np.zeros((n,) + batch_shape)
    for idx in range(n):
        downsampled[idx, ...] = features[int(length * idx / n):int(length * (idx + 1) / n), ...].mean(axis=0)
    return downsampled


def get_downsampler(n):
    return lambda features: downsampler(features=features, n=n)


def waveform_visualiser(*, features, position, app, update=True, **kwargs):
    features = WebApp.check_features(features[0], n=2)
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


def heatmap_visualiser(*, features, position, app, update=False, express=True, **kwargs):
    features = WebApp.check_features(features)
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


def advanced_chroma_visualiser_fast(*, features, position, app, **kwargs):
    features = WebApp.check_features(features)
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


def single_fourier(*, features, position, app, component, **kwargs):
    features = WebApp.check_features(features)
    features[1] *= rad_to_deg
    fig = px.line_polar(r=features[0, :, component], theta=features[1, :, component])
    idx =  WebApp.position_idx(position, n=features.shape[1])
    fig.add_trace(go.Scatterpolar(
        r=features[0, idx:idx + 1, component],
        theta=features[1, idx:idx + 1, component],
        mode='markers',
        marker=dict(size=10, color='red')
    ))
    return app.update_figure_layout(fig, **kwargs)


def fourier_visualiser(*, features, position, app, binary_profiles=False, incl=None, **kwargs):
    features = WebApp.check_features(features)
    features[1] *= rad_to_deg
    labels = [f"{nth} Coefficient" for nth in ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th']]
    specs = [[{'type': 'polar'} for _ in range(3)] for _ in range(2)]
    specs[1][2] = {}
    fig = make_subplots(rows=2, cols=3, start_cell="top-left", specs=specs,
                        # subplot_titles=labels
                        )
    idx = WebApp.position_idx(position, n=features.shape[1])
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
        angularaxis = {}
        if component == 5:
            if binary_profiles:
                # key signatures represented by binary profiles
                profiles = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]) / 7
                profiles = transpose_profiles(profiles)  # (trans, pitch)
                fourier_profiles = fourier_features(features=[profiles])  # (mag/phase, trans, component)
                fig.add_trace(go.Scatterpolar(
                    hoverinfo='skip',
                    r=fourier_profiles[0, :, component],
                    theta=fourier_profiles[1, :, component] * rad_to_deg,
                    mode='markers',
                    marker=dict(size=4, color='black'),
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
                fourier_profiles = fourier_features(features=[WebApp.major_minor_profiles])  # (mag/phase, mode, trans, component)
                fig.add_trace(go.Scatterpolar(
                    hoverinfo='skip',
                    r=fourier_profiles[0, :, :, component].flatten(),
                    theta=fourier_profiles[1, :, :, component].flatten() * rad_to_deg,
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    showlegend=False,
                ), row=row, col=col)
                # set ticks
                pitch_names = [str(EnharmonicPitchClass(n)) for n in range(12)]
                ma_mi_phase = np.concatenate([fourier_profiles[1, 0, :, component], fourier_profiles[1, 1, :, component]])
                angularaxis = dict(
                    tickmode='array',
                    tickvals=np.mod(ma_mi_phase * rad_to_deg, 360),
                    ticktext=np.array([[f"{n}ma", f"{n}mi"] for n in pitch_names]).T.flatten()
                )
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
    return app.update_figure_layout(fig, **kwargs)


def circle_of_fifths_visualiser(*, features, position, app, ticks="binary", **kwargs):
    features = WebApp.check_features(features)
    features[1] *= rad_to_deg
    idx = WebApp.position_idx(position, n=features.shape[1])
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
    profiles = KeyEstimator.profiles['albrecht']
    profiles = np.array([profiles['major'], profiles['minor']]).T  # (pitch, mode)
    profiles = transpose_profiles(profiles)  # (trans, pitch, mode)
    profiles = np.moveaxis(profiles, 2, 0)  # (mode, trans, pitch)
    fourier_profiles = fourier_features(features=[profiles])  # (mag/phase, mode, trans, component)
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


def tonnetz_visualiser(*, features, position, app, unicode=True, **kwargs):
    features = WebApp.check_features(features, asfarray=False)
    pos_idx = WebApp.position_idx(position, features=features)
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


def spectral_dome(*, features, position, app, **kwargs):
    features = WebApp.check_features(features, n=2, asfarray=False)
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


def keyscape(*, features, position, app, dark=False, legend=True, marker_legend=False, **kwargs):
    # lighting for surfaces
    lighting = dict(ambient=1., diffuse=1., roughness=1., specular=1., fresnel=1.)
    # process features
    features = WebApp.check_features(features, n=2, asfarray=False)
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
        legend_colors = key_colors(app.major_minor_profiles[:, (np.arange(12) * 7) % 12, :])  # (mode, trans, RGB)
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
