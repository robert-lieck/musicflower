#  Copyright (c) 2022 Robert Lieck.

import unittest
import pathlib

import librosa
from numpy.testing import assert_array_equal

from musicflower.loader import load_corpus


class MyTestCase(unittest.TestCase):

    def test_load_corpus(self):
        file_path = pathlib.Path(__file__).parent.resolve() / "Prelude_No._1_BWV_846_in_C_Major.ogg"
        corpus, file_names = load_corpus(data=[file_path, file_path], n=10, use_cache=True, recompute_cache=True)
        corpus, file_names = load_corpus(data=[file_path, file_path], n=10, use_cache=True, recompute_cache=False)
        y, sr = librosa.load(file_path)
        corpus_, file_names_ = load_corpus(data=[(y, sr), (y, sr)], n=10)
        assert_array_equal(corpus, corpus_)
