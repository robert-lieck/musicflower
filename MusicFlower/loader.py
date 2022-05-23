#  Copyright (c) Robert Lieck 2022.

import os
import pickle
from typing import Iterable
from itertools import product

from MusicFlower import istarmap # patch Pool
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import scipy

import librosa
import librosa.display

import pitchscapes.reader as rd

from TriangularMap import TMap


def get_cache_file_name(file_name, n, remove_extension=False):
    if remove_extension:
        return os.path.splitext(file_name)[0] + f"_n{n}.pickle"
    else:
        return file_name + f"_n{n}.pickle"


def audio_scape(n_time_intervals, file_path, normalise=True, top_down=False):
    # get chroma
    raw_chroma = get_chroma(file_name=file_path)
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
    flattened /= flattened.sum(axis=1, keepdims=True)
    return flattened


def load_file(file_name: str, n: int, use_cache=False, recompute_cache=False, audio_ext=(".wav", ".mp3", ".ogg"),
              **kwargs) -> np.ndarray:
    """
    Reads a single file and computes the corresponding pitch class distribution.
    :rtype: np.ndarray
    :param file_name: path of file to read from
    :param n: resolution at which to sample the piece.
    :param sample_scape_kwargs: kwargs to pass to pitchscapes.reader.sample_scape of the pitchscapes library.
                   important arguments are: . normalize: boolean indicating whether to normalize the counts row-wise
                                            . prior_counts : integer indicating the prior counts when computing the pcd
    :return: np.ndarray of shape ((n * (n+1)) / 2 , 12)
    """
    # argument Checks
    if n < 1:
        raise ValueError('Resolution should be a positive integer bigger than 1')
    if file_name is None:
        raise ValueError('File_name is None')
    if not os.path.isfile(file_name):
        raise FileNotFoundError(f'File {file_name} was not found')
    if not os.access(file_name, os.R_OK):
        raise ValueError(f'File {file_name} could not be read. Please verify permissions.')
    if not os.path.isfile(file_name):
        raise ValueError(f'File {file_name} should be a file not a directory')

    # get cache file
    if use_cache:
        cache_file_name = get_cache_file_name(file_name, n)
        cache_file_exists = os.path.exists(cache_file_name)
    else:
        cache_file_name = None
        cache_file_exists = None

    # load from cache or compute from scratch
    if use_cache and cache_file_exists and not recompute_cache:
        with open(cache_file_name, 'rb') as cache_file:
            pitch_class_distribution, cached_kwargs = pickle.load(cache_file)
        if not cached_kwargs == kwargs:
            raise ValueError(f"provided sample_scape_kwargs are different from cache file:\n"
                             f"    provided: {kwargs}\n"
                             f"    cache file: {cached_kwargs}\n"
                             f"Use recompute_cache=True to recompute and overwrite")
    else:
        if file_name.endswith(audio_ext):
            pitch_class_distribution = audio_scape(n_time_intervals=n,
                                                   file_path=file_name,
                                                   **kwargs)
        else:
            pitch_class_distribution = rd.sample_scape(n_time_intervals=n,
                                                       file_path=file_name,
                                                       **kwargs)
        if use_cache:
            with open(cache_file_name, 'wb') as cache_file:
                pickle.dump((pitch_class_distribution, kwargs), cache_file)

    return pitch_class_distribution


def _parallel_load_file_wrapper(file_name, kwargs):
    """
    Take additional kwargs as single argument and unpack.
    """
    return file_name, load_file(file_name, **kwargs)


def load_corpus(file_names: Iterable[str], parallel=False, top_down=True, sort_func=lambda x: x, **kwargs) -> tuple[np.ndarray, list]:
    """
    Computes pitch scape distributions for a collection of files.
    :rtype: np.ndarray
    :param file_names: Iterable[str], an iterable of file paths to read in the corpus.
    :param n: resolution at which to sample the pieces.
    :param kwargs: kwargs to pass to the load_file function.
    :return: np.ndarray of shape(len(file_names) , n * (n+1) / 2, 12) corresponding to the pitchscape for each file.
    """
    if parallel:
        with Pool() as pool:
            scape_list = []
            for file_name, scape in tqdm(pool.istarmap(_parallel_load_file_wrapper, product(file_names, [kwargs])),
                                         total=len(file_names)):
                scape_list.append((file_name, scape[None]))
    else:
        scape_list = []
        for file_name in tqdm(file_names):
            scape = load_file(file_name=file_name, **kwargs)
            scape_list.append((file_name, scape[None]))

    sorted_scape_list = sorted(scape_list, key=lambda x: sort_func(x[0]))
    final_array = np.concatenate([x[1] for x in sorted_scape_list], axis=0)
    file_names = [x[0] for x in sorted_scape_list]

    if top_down:
        return final_array[:, TMap.get_reindex_top_down_from_start_end(TMap.n_from_size1d(final_array.shape[1])), ...],\
               file_names
    else:
        return final_array, \
               file_names


def get_chroma(file_name, cqt=False, normal=False, harm=False, filter=False, smooth=True, asdict=False,
               loader=librosa.load):
    # adapted from https://librosa.org/doc/main/auto_examples/plot_chroma.html
    ret = []  # collect return values
    # read file
    y, sr = loader(file_name)
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
        file_name=file_name,
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
        file_path='/home/lieck/data/Fabians_corpus/scores/Bach_JohannSebastian/210606-Prelude_No._1_BWV_846_in_C_Major.ogg')
