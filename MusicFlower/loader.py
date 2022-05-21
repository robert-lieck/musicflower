#  Copyright (c) Robert Lieck 2022.

import os
import pickle
from typing import Iterable
from itertools import product

from MusicFlower import istarmap # patch Pool
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

import pitchscapes.reader as rd

from TriangularMap import TMap


def get_cache_file_name(file_name, n):
    return os.path.splitext(file_name)[0] + f"_n{n}.pickle"


def load_file(file_name: str, n: int, use_cache=False, recompute_cache=False, **kwargs) -> np.ndarray:
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


def load_corpus(file_names: Iterable[str], parallel=False, top_down=True, sort_func=lambda x: x, **kwargs) -> np.ndarray:
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
            final_array = np.concatenate([x[1] for x in sorted(scape_list, key=lambda x: sort_func(x[0]))], axis=0)
    else:
        raise NotImplementedError
        # final_array = None
        # for file_name in tqdm(file_names):
        #     pcd = load_file(file_name=file_name, **kwargs)
        #     final_array = pcd if final_array is None else np.concatenate((final_array, pcd))

    if top_down:
        return final_array[:, TMap.get_reindex_top_down_from_start_end(TMap.n_from_size1d(final_array.shape[1])), ...]
    else:
        return final_array


def start_duration(n):
    start = []
    duration = []
    for idx in range(1, n + 1):
        start.append(np.arange(idx) / n)
        duration.append(np.ones(idx) - idx / n)
    return np.concatenate(start), np.concatenate(duration)