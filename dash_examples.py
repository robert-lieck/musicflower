#  Copyright (c) 2023 Robert Lieck.
"""
Interactive Plot
===========================

This is a basic example of how to create an interactive plot.
"""

# %%
# Dash Hello World
# ------------------------
#
# Launch a simple dash "Hello World" app.

from dash import Dash, html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64


# some parameters to config/change the app

pre_load_sound = True
n_points = 500
poll_interval_ms = 400

# version = "animation"
# version = "dash-animation"
version = "audio-sync"
# version = "store-frames"
# version = "sparse-updates"
# version = "sparse-updates-client-only"


# initialise the app
app = Dash(__name__, update_title=None)

# generate some dummy data for a plot
data = np.concatenate([np.random.normal(size=n_points)[:, None], np.linspace(0, 1, n_points)[:, None]], axis=1)
df = pd.DataFrame(data, columns=['value', 'time'])

# preload audio file via Python or assume it is present in asset folder
if pre_load_sound:
    sound_filename = 'test_sound.wav'
    encoded_sound = base64.b64encode(open(sound_filename, 'rb').read())
    audio_src = 'data:audio/mpeg;base64,{}'.format(encoded_sound.decode())
else:
    audio_src = app.get_asset_url('test_sound.wav')


# functions to plot the data and a marker at specified position
def plot_data():
    return go.Scatter(x=data[:, 1], y=data[:, 0], mode='lines')


def plot_marker(idx):
    return go.Scatter(x=data[idx:idx + 1, 1], y=data[idx:idx + 1, 0], mode='markers')


def compute_frame(idx=None, transition=True, show_data=True):
    if show_data:
        fig_data = [plot_data()]
    else:
        fig_data = [go.Scatter()]
    if idx is None:
        fig_data.append(go.Scatter())
    else:
        fig_data.append(plot_marker(idx))
    fig = go.Figure(data=fig_data)
    fig.update_xaxes(range=[0, 1])
    if transition:
        fig.update_layout(transition_duration=poll_interval_ms / 2)
    return fig


# different versions during development using different techniques
if version == "animation":
    # create a figure with frames used in an animation with a slider
    # - this also works without Dash
    # - frames are stored on the client side, which allows for fast client-side updates
    # - traces present in multiple/all frames need to be duplicated and re-rendered, which makes it slow for large plots
    fig = compute_frame()
    frames = [go.Frame(data=compute_frame(idx).data, name=f'{idx}') for idx in range(n_points)]
    fig.update_layout(sliders=[dict(steps=[dict(method='animate',
                                                   args=[[f'{k}'],
                                                         dict(mode='immediate',
                                                              frame=dict(duration=300),
                                                              transition=dict(duration=0))],
                                                   label=f"{k/n_points}")
                                              for k in range(n_points)])])
    fig.update(frames=frames)

    app.layout = html.Div([
        dcc.Graph(figure=fig, id='graphID'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
    ])
elif version == "dash-animation":
    # implement the 'animation' via Dash using a slider and sever-side callback
    # - this re-generates the plot on each change of the slider, which is slow
    @callback(
        Output(component_id='graphID', component_property='figure'),
        Input(component_id='sliderID', component_property='value')
    )
    def update_graph(value):
        return compute_frame(value)

    # also synchronise the audio element with the slider using a client-side callback
    app.clientside_callback(
        """
        function(min, max, value) {
            const audio = document.getElementById("audioID");
            audio.currentTime = audio.duration * value / (max - min);
            return value;
        }
        """,
        Output(component_id='dummy', component_property='value'),  # dummy output required
        Input(component_id='sliderID', component_property='min'),
        Input(component_id='sliderID', component_property='max'),
        Input(component_id='sliderID', component_property='value'),
        prevent_initial_call=True  # prevent error from uninitialised audio element at startup
    )

    app.layout = html.Div([
        dcc.Graph(figure={}, id='graphID'),
        dcc.Slider(0, n_points, 1, value=0, id='sliderID', updatemode='drag'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
        dcc.Input(id='dummy', value="", style={'display': 'none'}),
    ])
elif version == "audio-sync":
    # use the audio element instead of a slider
    # - this requires polling the audio position as no corresponding callback is implemented in Dash
    # - the result is stored in dummy element, which is used to trigger an update callback for the graph
    # - plots are still re-generated on server-side, so still slow
    # - using two dummy divs to store/track changes and only update if value has actually changed

    # poll and store audio position
    app.clientside_callback(
        """
        function(value) {
            const audio = document.getElementById("audioID");
            return audio.currentTime / audio.duration;
        }
        """,
        Output(component_id='new-audio-position', component_property='value'),
        Input(component_id='audio-sync', component_property='n_intervals'),
        prevent_initial_call=True
    )

    # update plot if value changed
    @callback(
        Output(component_id='graphID', component_property='figure'),
        Output(component_id='old-audio-position', component_property='value'),
        Input(component_id='new-audio-position', component_property='value'),
        State(component_id='old-audio-position', component_property='value'),
        prevent_initial_call=True
    )
    def update_graph(new_pos, old_pos):
        idx = int(new_pos * (n_points - 1))
        if new_pos == old_pos:
            # only recompute if value has changed
            raise PreventUpdate
        else:
            return compute_frame(idx), new_pos

    app.layout = html.Div([
        dcc.Graph(figure={}, id='graphID'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
        dcc.Input(id='old-audio-position', value="", style={'display': 'none'}),
        dcc.Input(id='new-audio-position', value="", style={'display': 'none'}),
        dcc.Interval(id='audio-sync', interval=poll_interval_ms),
    ])
elif version == "store-frames":
    # pre-compute frames and store on client side to speed up animation
    fig = compute_frame()
    frames = [go.Frame(data=compute_frame(idx).data, name=f'{idx}') for idx in range(n_points)]
    fig.update(frames=frames)

    # poll and store audio position
    app.clientside_callback(
        """
        function(dummy) {
            const audio = document.getElementById("audioID");
            return audio.currentTime / audio.duration;
        }
        """,
        Output(component_id='new-audio-position', component_property='value'),
        Input(component_id='audio-sync', component_property='n_intervals'),
        prevent_initial_call=True
    )

    # pass on new value if changed
    @callback(
        Output(component_id='old-audio-position', component_property='value'),
        Input(component_id='new-audio-position', component_property='value'),
        State(component_id='old-audio-position', component_property='value'),
        prevent_initial_call=True
    )
    def update_graph(new_pos, old_pos):
        if new_pos == old_pos:
            # only recompute if value has changed
            raise PreventUpdate
        else:
            return new_pos

    # update graph on change
    app.clientside_callback(
        """
        function(audio_position, n_frames, frames){
            const idx = parseInt((n_frames - 1) * audio_position)
            return {'data': frames[idx]['data'], 'layout': frames[idx]['layout']};
        }
        """,
        Output("graphID", "figure"),
        Input(component_id='old-audio-position', component_property='value'),
        State("n-frames", "data"),
        State("frames", "data"),
    )

    app.layout = html.Div([
        dcc.Graph(figure={}, id='graphID'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
        dcc.Input(id='old-audio-position', value='', style={'display': 'none'}),
        dcc.Input(id='new-audio-position', value='', style={'display': 'none'}),
        dcc.Interval(id='audio-sync', interval=poll_interval_ms),
        dcc.Store(id='frames', data=fig['frames']),
        dcc.Store(id='n-frames', data=n_points),
    ])
elif version == "sparse-updates":
    # pre-compute frames and store on client side to speed up animation

    # poll and store audio position
    app.clientside_callback(
        """
        function(dummy) {
            const audio = document.getElementById("audioID");
            return audio.currentTime / audio.duration;
        }
        """,
        Output(component_id='new-audio-position', component_property='value'),
        Input(component_id='audio-sync', component_property='n_intervals'),
        prevent_initial_call=True
    )

    # pass on new value if changed
    @callback(
        Output(component_id='old-audio-position', component_property='value'),
        Input(component_id='new-audio-position', component_property='value'),
        State(component_id='old-audio-position', component_property='value'),
        prevent_initial_call=True
    )
    def update_graph(new_pos, old_pos):
        if new_pos == old_pos:
            # only recompute if value has changed
            raise PreventUpdate
        else:
            return new_pos

    # update graph on change
    app.clientside_callback(
        """
        function(audio_position, n_frames, frames, figure){
            const idx = parseInt((n_frames - 1) * audio_position)
            return [{x: [frames[idx]['data'][1]['x']], y: [frames[idx]['data'][1]['y']]}, [1], 1];
        }
        """,
        Output("graphID", "extendData"),
        Input(component_id='old-audio-position', component_property='value'),
        State("n-frames", "data"),
        State("frames", "data"),
        State("graphID", "figure")
    )

    fig = compute_frame(0)
    frames = [go.Frame(data=compute_frame(idx, show_data=False).data, name=f'{idx}') for idx in range(n_points)]

    app.layout = html.Div([
        dcc.Graph(figure=fig, id='graphID'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
        dcc.Input(id='old-audio-position', value='', style={'display': 'none'}),
        dcc.Input(id='new-audio-position', value='', style={'display': 'none'}),
        dcc.Interval(id='audio-sync', interval=poll_interval_ms),
        dcc.Store(id='frames', data=frames),
        dcc.Store(id='n-frames', data=n_points),
    ])
elif version == "sparse-updates-client-only":
    # pre-compute frames and store on client side to speed up animation
    # use only client-side callbacks
    # STILL: cannot be saved as static page

    # poll and store audio position
    app.clientside_callback(
        """
        function(dummy, n_frames, frames, figure) {
            const audio = document.getElementById("audioID");
            const audio_position = audio.currentTime / audio.duration;
            const idx = parseInt((n_frames - 1) * audio_position)
            return [{x: [frames[idx]['data'][1]['x']], y: [frames[idx]['data'][1]['y']]}, [1], 1];
        }
        """,
        Output("graphID", "extendData"),
        Input(component_id='audio-sync', component_property='n_intervals'),
        State("n-frames", "data"),
        State("frames", "data"),
        State("graphID", "figure"),
        prevent_initial_call=True
    )

    fig = compute_frame(0)
    frames = [go.Frame(data=compute_frame(idx, show_data=False).data, name=f'{idx}') for idx in range(n_points)]

    app.layout = html.Div([
        dcc.Graph(figure=fig, id='graphID'),
        html.Audio(src=audio_src, controls=True, id="audioID", style={'width': '100%'}),
        dcc.Interval(id='audio-sync', interval=poll_interval_ms),
        dcc.Store(id='frames', data=frames),
        dcc.Store(id='n-frames', data=n_points),
    ])
else:
    raise RuntimeError(f"Unknown version '{version}'")


# run the server
if __name__ == '__main__':
    app.run_server(debug=True)
