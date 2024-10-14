"""
Web App Example
===============

This is a basic example for creating a MusicFlower web app with a custom visualiser.
"""

# %%
# Custom Visualiser
# -----------------
#
# Creating a web app with a custom visualiser is as easy as defining a single callback function

import plotly.graph_objects as go

def my_custom_visualiser(*, features, position, **kwargs):
    return go.Figure(data=[go.Bar(
        y=features[0][round(position * (len(features[0]) - 1))]
    )])


# %%
# We can now register this function as a visualiser and start up the web app

from musicflower.webapp import WebApp

WebApp() \
    .use_chroma_features() \
    .register_visualiser('Chroma Bars', ['chroma-features'], my_custom_visualiser) \
    .init() \
    # .run()  # (uncomment this line!)

# %%
# A slightly more elaborate version of the visualiser would be as follows

from pitchtypes import EnharmonicPitchClass

def my_custom_visualiser(*, features, position, **kwargs):
    features = WebApp.check_features(features)
    position = WebApp.position_idx(position, features=features)
    data = features[position]
    fig = go.Figure(data=[go.Bar(
        x=[str(EnharmonicPitchClass(i)) for i in range(12)],
        y=data
    )])
    fig.update_yaxes(range=[0, 1])
    return WebApp.update_figure_layout(fig)

# %%
# And the app can be set up with additional parameters

app = WebApp(verbose=True)  # print information about callbacks
app.use_chroma_features(200)  # maximum time resolution
app.register_visualiser('Chroma Bars', ['chroma-features'], my_custom_visualiser)
app.init(
    figure_height=500,  # specify figure dimensions
    # audio_file="/path/to/initial/audio/file.mp3",  # audio file to load at start up
)
# app.run(debug=True)  # run app in debug mode (uncomment this line!)
