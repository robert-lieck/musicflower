# MusicFlower

[![tests](https://github.com/robert-lieck/musicflower/actions/workflows/tests.yml/badge.svg)](https://github.com/robert-lieck/musicflower/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/robert-lieck/musicflower/branch/main/graph/badge.svg?token=D3G3NI02UB)](https://codecov.io/gh/robert-lieck/musicflower)

![build](https://github.com/robert-lieck/musicflower/workflows/build/badge.svg)
[![PyPI version](https://badge.fury.io/py/musicflower.svg)](https://badge.fury.io/py/musicflower)

[![doc](https://github.com/robert-lieck/musicflower/actions/workflows/doc.yml/badge.svg)](https://github.com/robert-lieck/musicflower/actions/workflows/doc.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

For more information, have a look at the [documentation](https://robert-lieck.github.io/musicflower/).

![logo](doc/logo_96.png)

This is an anonymised version of the \<library> Python package source code with instructions on how to start up the web app for testing.

## Setup

Start from a clean Python environment by running the following in a terminal (we assume you are using anaconda; the package was tested with Python version 3.9)

```
$ conda create --name testenv python=3.9
```

and activate it

```
$ conda activate testenv
```

Install all requirements via (assuming your current working directory is the one this file is in)

```
$ pip install -r requirements.txt
```

## Starting the app

You can now start the web app by running

```
$ python -m library
```

And open it in your web browser on, typically at http://127.0.0.1:8050/ (the URL is also printed in the terminal).

## Problems with MP3 Files

If you experience problems with loading mp3 files, you may need to install additional codecs on your system, which are required by `librosa` for reading the audio files (see [here](https://github.com/librosa/librosa#audioread-and-mp3-support)).
