.. documentation master file; adapt to your liking (but should at least contain the root `toctree` directive).

Welcome to MusicFlower's documentation!
=======================================

MusicFlower is a Python package designed to make it as easy as possible to create dynamic visualisations synchronised
with audio playback. It is built on top of Plotly/Dash and runs a local web app for interactve display (which can of
course also be deployed online if desired).

MusicFlower comes with several pre-implemented visualisations and can easily be extended by registering a callback
function that returns a static Plotly figure, which is then synchronised with audio playback.

Have a look at the :doc:`auto_examples/index` for some examples.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   auto_examples/index.rst
   api_summary
