#  Copyright (c) Robert Lieck 2022.

import os
import pickle
from typing import Iterable, Tuple, Union, Dict
from itertools import product

from musicflower import istarmap # patch Pool
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import scipy

import librosa
import librosa.display

import pitchscapes.reader as rd

from triangularmap import TMap


def get_cache_file_path(file_path: str, n: int, remove_extension: bool = False):
    """
    For a given file path and resolution, return the associated cache file path, which corresponds to the original file
    path appended with `_<n>.pickle`, where `<n>` is replaced with the actual resolution.

    :param file_path: path to the original file
    :param n: resolution of the pitch scape
    :param remove_extension: remove the original file extension before appending `_<n>.pickle` (if the same file is used
     with different extensions, for instance, \*.mxl for a MusicXML file and \*.ogg for an associated audio file,
     removing the extension leads to name conflicts for the cache files)
    :return: path to the cache file
    """
    if remove_extension:
        return os.path.splitext(file_path)[0] + f"_n{n}.pickle"
    else:
        return file_path + f"_n{n}.pickle"


def audio_scape(n_time_intervals: int, data: Union[str, Tuple[np.ndarray, int]], normalise: bool = True,
                top_down: bool = False, **kwargs) -> np.ndarray:
    """
    Compute a pitch scape from audio data.

    :param n_time_intervals: number of time intervals
    :param data: path to file or a tuple with audio data (passed on to :func:`~musicflower.loader.get_chroma`)
    :param normalise: normalise the pitch class distributions
    :param top_down: use top-down order
    :param kwargs: kwargs passed on to the :func:`~musicflower.loader.get_chroma` function
    :return: array with pitch scape
    """
    # get chroma
    raw_chroma = get_chroma(data=data, **kwargs)
    n_bins = raw_chroma.shape[1]
    # sum over specified time intervals
    chroma = np.zeros((n_time_intervals, 12))
    for idx in range(n_time_intervals):
        start = int(round(idx / n_time_intervals * n_bins))
        end = int(round((idx + 1) / n_time_intervals * n_bins))
        chroma[idx, :] = raw_chroma[:, start:end].sum(axis=1)
    # recursively build scape
    scape = [chroma]
    current_level = chroma
    for idx in range(n_time_intervals):
        next_level = current_level[:-1] + chroma[idx + 1:]
        scape.append(next_level)
        current_level = next_level
    # reverse (because was constructed bottom-up not top-down) and concatenate
    flattened = np.concatenate(list(reversed(scape)))
    # change to ordering used by pitchscapes library
    if not top_down:
        flattened = flattened[TMap.get_reindex_start_end_from_top_down(TMap.n_from_size1d(flattened.shape[0])), ...]
    # normalise
    if normalise:
        flattened /= flattened.sum(axis=1, keepdims=True)
    return flattened


def load_file(data: str, n: int, use_cache=False, recompute_cache=False, audio=None,
              audio_ext=(".wav", ".mp3", ".ogg"), top_down=True, **kwargs) -> np.ndarray:
    """
    Reads a single file and computes its pitch scape with resolution *n*.

    :param data: path to file or a tuple with audio data (see :func:`~musicflower.loader.get_chroma`)
    :param n: resolution of pitch scape, i.e., the number of equally-sized time intervals to split the piece into
    :param use_cache: whether to use/reuse cached results; these are stored in cache files specific for each resolution
     *n* (see :meth:`~musicflower.loader.get_cache_file_path`). The **kwargs** are stored with the cache files and
     checked for consistency; an error is raised if they do not match the provided **kwargs**. If **use_cache** is
     `True` and a cache file exists, the cached result is loaded (after checking **kwargs** for consistency) and
     returned; if the cache file does not exist, the result is computed and stored in a newly created cache file. This
     behaviour can be changed by using **recompute_cache**.
    :param recompute_cache: if caching is used, always recompute the result and overwrite potentially existing cache
     files
    :param audio: specifies that this is an audio file (**audio_ext** is ignored)
    :param audio_ext: assumes all files with an extension in this list to be audio files, all other files to be symbolic
    :param top_down: use the top-down ordering for flattening the triangular map (as used by the :class:`TMap` class);
     if `False` the start-to-end convention from the `pitchscapes` library is used
    :param kwargs: kwargs to passed to the :meth:`pitchscapes.reader.sample_scape` function of the pitchscapes library
     (for symbolic data) or the :func:`~musicflower.loader.audio_scape` function (for audio data);
     by default `normalise=True` is injected into kwargs, but this is overwritten by an explicitly specified value
    :rtype: np.ndarray
    :return: array of shape (k , 12), where :math:`k=n(n+1)/2`
    """
    # normalise by default
    kwargs = {**dict(normalise=True), **kwargs}
    # argument Checks
    if n < 1:
        raise ValueError('Resolution should be a positive integer bigger than 1')
    if data is None:
        raise ValueError('data is None')
    if not isinstance(data, tuple):
        if not os.path.isfile(data):
            print(os.getcwd())
            raise FileNotFoundError(f'File {data} was not found')
        if not os.access(data, os.R_OK):
            raise ValueError(f'File {data} could not be read. Please verify permissions.')
        if not os.path.isfile(data):
            raise ValueError(f'File {data} is not a file')

    # get cache file
    if use_cache:
        cache_file_name = get_cache_file_path(data, n)
        cache_file_exists = os.path.exists(cache_file_name)
    else:
        cache_file_name = None
        cache_file_exists = None

    # load from cache or compute from scratch
    if use_cache and cache_file_exists and not recompute_cache:
        with open(cache_file_name, 'rb') as cache_file:
            pdc, cached_kwargs = pickle.load(cache_file)
        if not cached_kwargs == kwargs:
            raise ValueError(f"provided sample_scape_kwargs are different from cache file:\n"
                             f"    provided: {kwargs}\n"
                             f"    cache file: {cached_kwargs}\n"
                             f"Use recompute_cache=True to recompute and overwrite")
    else:
        if audio or isinstance(data, tuple) or (audio is None and data.endswith(audio_ext)):
            pdc = audio_scape(n_time_intervals=n,
                              data=data,
                              **kwargs)
        else:
            pdc = rd.sample_scape(n_time_intervals=n,
                                  file_path=data,
                                  **kwargs)
        if use_cache:
            with open(cache_file_name, 'wb') as cache_file:
                pickle.dump((pdc, kwargs), cache_file)

    if top_down:
        return pdc[TMap.get_reindex_top_down_from_start_end(TMap.n_from_size1d(pdc.shape[0])), ...]
    else:
        return pdc


def _parallel_load_file_wrapper(file_name, kwargs):
    """
    Wrapper for parallelisation; takes kwargs provided as dict and unpacks them when calling
    :meth:`~musicflower.loader.load_file`.
    """
    return file_name, load_file(file_name, **kwargs)


def load_corpus(data: Iterable, n: int, parallel: bool = False, sort_func: callable = lambda x: x,
                **kwargs) -> Tuple[np.ndarray, list]:
    """
    This is essentially a wrapper for parallelisation around the :meth:`~musicflower.loader.load_file` function, which
    computes pitch scapes for a set of files.

    :param data: an iterable of file paths or tuples with audio data (see :func:`~musicflower.loader.get_chroma`)
    :param n: resolution of pitch scape, i.e., the number of equally-sized time intervals to split the piece into
    :param parallel: parallelise loading
    :param sort_func: function that takes a file path and returns a key for sorting the result
    :param kwargs: kwargs passed to the :meth:`~musicflower.loader.load_file` function (Note: this *has* to include
    :rtype: np.ndarray
    :return: array of shape (k, l, 12) where k is the number of files and :math:`l=n(n+1)/2` is the number of points in
     a pitch scape of resolution :math:`n`.
    """
    # add resolution to kwargs
    kwargs['n'] = n
    # process in parallel or sequentially
    if parallel:
        with Pool() as pool:
            scape_list = []
            for file_name, scape in tqdm(pool.istarmap(_parallel_load_file_wrapper, product(data, [kwargs])),
                                         total=len(data)):
                scape_list.append((file_name, scape[None]))
    else:
        scape_list = []
        for file_name in tqdm(data):
            scape = load_file(data=file_name, **kwargs)
            scape_list.append((file_name, scape[None]))

    sorted_scape_list = sorted(scape_list, key=lambda x: sort_func(x[0]))
    final_array = np.concatenate([x[1] for x in sorted_scape_list], axis=0)
    data = [x[0] for x in sorted_scape_list]

    return final_array, data


def get_chroma(data: Union[str, Tuple[np.ndarray, int]], cqt: bool = False, normal: bool = False, harm: bool = False,
               filter: bool = False, smooth: bool = True, asdict: bool = False, loader=librosa.load
               ) -> Union[Tuple[np.ndarray], Dict[str, np.ndarray]]:
    """
    Get chroma features from audio data.

    :param data: a path to an audio file, which is loaded using librosa's
     `load <https://librosa.org/doc/main/generated/librosa.load.html>`_ function, or a tuple ``(y, sr)`` that is
     of the same type as returned by the load function, i.e. ``y``: an audio time series as np.ndarray and ``sr`` the
     sampling rate
    :param cqt: return constant-Q transform
    :param normal: return "normal" chroma features
    :param harm: filter harmonics before computing chroma features
    :param filter: apply non-local filtering to chroma features
    :param smooth: apply a horizontal median filter (after non-local filtering)
    :param asdict: return results as a dict (keys are 'cqt', 'normal', 'harm', 'filter', 'smooth')
    :param loader: the loader to use (defaults to librosa.load)
    :return: tuple or dict with results (cqt/normal/harm/filter/smooth as specified)
    """
    # adapted from https://librosa.org/doc/main/auto_examples/plot_chroma.html
    ret = []  # collect return values
    # read file
    if isinstance(data, tuple):
        y, sr = data
    else:
        y, sr = loader(data)
    # CQT matrix
    if cqt:
        ret.append(('cqt', np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=12 * 3, n_bins=7 * 12 * 3))))
    # original chroma
    if normal:
        ret.append(('normal', librosa.feature.chroma_cqt(y=y, sr=sr)))
    # improvements
    if harm or filter or smooth:
        # isolating harmonic component
        y_harm = librosa.effects.harmonic(y=y, margin=8)
        chroma_harm = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
        if harm:
            ret.append(('harm', chroma_harm))
        if filter or smooth:
            # non-local filtering
            chroma_filter = np.minimum(chroma_harm,
                                       librosa.decompose.nn_filter(chroma_harm,
                                                                   aggregate=np.median,
                                                                   metric='cosine'))
            if filter:
                ret.append(('filter', chroma_filter))
            # horizontal median filter
            if smooth:
                ret.append(('smooth', scipy.ndimage.median_filter(chroma_filter, size=(1, 9))))
    # return
    if asdict:
        return dict(ret)
    else:
        if len(ret) == 1:
            return ret[0][1]
        else:
            return tuple([x[1] for x in ret])


def plot_chroma_comparison(file_name, start, end):
    import matplotlib.pyplot as plt
    cqt, normal, smooth = get_chroma(
        data=file_name,
        cqt=True, normal=True, smooth=True)

    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([start, end])))])
    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(cqt, ref=np.max)[idx],
                             y_axis='cqt_note', x_axis='time',
                             bins_per_octave=12 * 3, ax=ax[0])
    ax[0].set(ylabel='CQT')
    ax[0].label_outer()
    librosa.display.specshow(normal[idx], y_axis='chroma', x_axis='time', ax=ax[1])
    ax[1].set(ylabel='Default chroma')
    ax[1].label_outer()
    librosa.display.specshow(smooth[idx], y_axis='chroma', x_axis='time', ax=ax[2])
    ax[2].set(ylabel='Processed')
    plt.show()


if __name__ == "__main__":
    audio_scape(
        n_time_intervals=10,
        data='/home/lieck/data/Fabians_corpus/scores/Bach_JohannSebastian/210606-Prelude_No._1_BWV_846_in_C_Major.ogg')
