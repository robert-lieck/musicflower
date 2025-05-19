"""
WebApp Example
==============

This is a basic example for creating a MusicFlower web app with a custom visualiser.
"""

# %%
# Default Version
# ---------------
#
# The default version of the web app, which you can run with `python -m musicflower`, uses the following startup script

from musicflower.webapp import WebApp
import musicflower.features as f
import musicflower.visualisers as v

app = WebApp(verbose=True)                      # print additional info about registered features, callbacks etc.
f.use_chroma_features(app, n=200)               # max temporal resolution of 200
v.use_fourier_visualiser(app)                   # register visualiser and required features
v.use_keyscape_visualiser(app)                  # ...
v.use_tonnetz_visualiser(app)                   # ...
v.use_spectral_dome_visualiser(app)             # ...
app.init(figure_width=1500, figure_height=800)  # larger figures
# app.run()                                     # start up the web app (uncomment to actually run!)

# %%
# The `use_...` functions are convenience functions for features and visualisers to automatically register the
# required dependencies. A more explicit version of the startup script would look like this

app = WebApp(verbose=True)

# define a downsampled version of chroma features using feature remappers
def chroma_features_n(*, audio, app, n=200):
    cf = f.chroma_features(audio=audio, app=app, normalised=False)
    cf = f.downsampler(features=[cf], app=app, n=n)
    cf = f.normaliser(features=[cf], app=app)
    return cf

# features
app.register_feature_extractor(name='chroma-features',
                               func=chroma_features_n)
app.register_feature_remapper(remapper_name='fourier-features',
                              feature_names=['chroma-features'],
                              func=f.fourier_features)
app.register_feature_remapper(remapper_name='chroma-scape-features',
                              feature_names=['chroma-features'],
                              func=f.chroma_scape_features)
app.register_feature_remapper(remapper_name='fourier-scape-features',
                              feature_names=['chroma-scape-features'],
                              func=f.fourier_features)

# visualisers
app.register_visualiser(visualiser_name='Fourier Coefficients',
                        feature_names=['fourier-features'],
                        func=v.fourier_visualiser)
app.register_visualiser(visualiser_name='Keyscape',
                        feature_names=['fourier-scape-features',
                                       'chroma-scape-features'],
                        func=v.keyscape_visualiser)
app.register_visualiser(visualiser_name='Tonnetz',
                        feature_names=['chroma-features'],
                        func=v.tonnetz_visualiser)
app.register_visualiser(visualiser_name='Spectral Dome',
                        feature_names=['fourier-scape-features',
                                       'chroma-scape-features'],
                        func=v.spectral_dome_visualiser)

app.init(figure_width=1500, figure_height=800)
# app.run()  # uncomment to actually run


# %%
# Custom Visualiser
# -----------------
#
# Creating your own a web app with a custom visualiser is as easy as defining a single callback function

import plotly.graph_objects as go

def my_custom_visualiser(*, features, position, **kwargs):
    return go.Figure(data=[go.Bar(
        y=features[0][round(position * (len(features[0]) - 1))]
    )])


# %%
# We can now register this function as a visualiser and start up the web app with some additional parameters to
# facilitate debugging

from musicflower.webapp import WebApp

app = WebApp(verbose=True)
f.use_chroma_features(app)
app.register_visualiser('Chroma Bars', ['chroma-features'], my_custom_visualiser)
app.init(
    # audio_file="/path/to/audio/file.mp3"  # automatically pre-loading an audio file can be convenient for debugging
)
# app.run(debug=True)  # run the app in debug mode to see exceptions raised in callback functions

# %%
# A slightly more elaborate version of the visualiser could look as follows

from pitchtypes import EnharmonicPitchClass

def my_custom_visualiser(*, features, position, app=WebApp, **kwargs):
    features = app.check_features(features)
    position = app.position_idx(position, features=features)
    data = features[position]
    fig = go.Figure(data=[go.Bar(
        x=[str(EnharmonicPitchClass(i)) for i in range(12)],
        y=data
    )])
    fig.update_yaxes(range=[0, 1])
    return app.update_figure_layout(fig)


app = WebApp(verbose=True)
f.use_chroma_features(app, 200)
app.register_visualiser('Chroma Bars', ['chroma-features'], my_custom_visualiser)
app.init()
# app.run(debug=True)
