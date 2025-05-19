#  Copyright (c) 2025 Robert Lieck.

import os
from tempfile import NamedTemporaryFile
from warnings import warn

import numpy as np
import librosa

from musicflower.loader import get_chroma, audio_scape
from musicflower.util import get_fourier_component, feature_exists
from musicflower.webapp import WebApp


def none_feature(*, audio, app=WebApp):
    return []


def waveform_feature(*, audio, app=WebApp, use_real_file=True):
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


def spectrogram_features(*, audio, app=WebApp):
    y, sr = waveform_feature(audio=audio, app=app, use_real_file=True)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db.T


def chroma_features(*, audio, app=WebApp, normalised=True):
    y, sr = waveform_feature(audio=audio, app=app, use_real_file=True)
    chroma = get_chroma(
        data=(y, sr),
        # hop_length=512,  # default
        # hop_length=2048,
    ).T
    if normalised:
        chroma = normaliser(features=[chroma], app=app)
    return chroma


def use_chroma_features(app, n=None, name='chroma-features', ignore_existing=False):
    if ignore_existing and feature_exists(app, name):
        return
    if n is None:
        app.register_feature_extractor(name, chroma_features)
    else:
        def chroma_features_n(*, audio, app, n=n):
            f = chroma_features(audio=audio, app=app, normalised=False)
            f = downsampler(features=[f], app=app, n=n)
            f = normaliser(features=[f], app=app)
            return f

        app.register_feature_extractor(name, chroma_features_n)


def chroma_scape_features(*, features, app=WebApp):
    features = app.check_features(features)
    return audio_scape(n_time_intervals=features.shape[0], raw_chroma=features.T, top_down=True)


def use_chroma_scape_features(app, name='chroma-scape-features', chroma_name=None, ignore_existing=False):
    if ignore_existing and feature_exists(app, name):
        return
    if chroma_name is None:
        chroma_name = 'chroma-features'
        if not feature_exists(app, chroma_name):
            use_chroma_features(app, name=chroma_name)
    app.register_feature_remapper(name, [chroma_name], chroma_scape_features)


def normaliser(*, features, app=WebApp, inplace=True, axis=-1):
    features = app.check_features(features)
    if inplace:
        features /= features.sum(axis=axis, keepdims=True)
        return features
    else:
        return features / features.sum(axis=axis, keepdims=True)


def fourier_features(*, features, app=WebApp):
    """
    Compute the amplitude and phase of all Fourier components using `get_fourier_component`.
    """
    features = app.check_features(features)
    return get_fourier_component(features)


def use_fourier_features(app, name='fourier-features', chroma_name=None, ignore_existing=False):
    if ignore_existing and feature_exists(app, name):
        return
    if chroma_name is None:
        chroma_name = 'chroma-features'
        if not feature_exists(app, chroma_name):
            use_chroma_features(app, name=chroma_name)
    app.register_feature_remapper(name, [chroma_name], fourier_features)


def use_fourier_scape_features(app, name='fourier-scape-features', chroma_name=None, ignore_existing=False):
    if ignore_existing and feature_exists(app, name):
        return
    if chroma_name is None:
        chroma_name = 'chroma-scape-features'
        if not feature_exists(app, chroma_name):
            use_chroma_features(app, name=chroma_name)
    app.register_feature_remapper(name, [chroma_name], fourier_features)


def downsampler(*, features, app=WebApp, n):
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


def get_downsampler(n):
    return lambda features: downsampler(features=features, n=n)
