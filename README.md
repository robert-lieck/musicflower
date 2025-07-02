# MusicFlower

<img src="doc/logo_96.png" alt="logo" style="zoom: 25%;" />

[![tests](https://github.com/robert-lieck/musicflower/actions/workflows/tests.yml/badge.svg)](https://github.com/robert-lieck/musicflower/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/robert-lieck/musicflower/branch/main/graph/badge.svg?token=D3G3NI02UB)](https://codecov.io/gh/robert-lieck/musicflower)

![build](https://github.com/robert-lieck/musicflower/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/musicflower.svg)](https://badge.fury.io/py/musicflower)

[![doc](https://github.com/robert-lieck/musicflower/actions/workflows/doc.yml/badge.svg)](https://github.com/robert-lieck/musicflower/actions/workflows/doc.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

MusicFlower provides a web app and python framework for interactive music visualisation. See below for a quick start and have a look at the [documentation](https://robert-lieck.github.io/musicflower/) for more information and examples.

## Quickstart

Starting from a clean Python environment, install MusicFlower

```
$ pip install musicflower
```

You can now start the web app by running

```
$ python -m musicflower
```

and open it in your web browser (typically) at http://127.0.0.1:8050/ (the URL is also printed in the terminal). If you would like to customise the app, you can modify the default startup scrip (e.g. to exclude visualisers you do not need to make the app more reactive, have a look at [this example](https://robert-lieck.github.io/musicflower/auto_examples/plot_webapp.html) for more details)

```python
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
app.run()
```

## Problems with MP3 Files

If you experience problems with loading mp3 files, you may need to install additional codecs on your system, which are required by `librosa` for reading the audio files (see [here](https://github.com/librosa/librosa#audioread-and-mp3-support)).
