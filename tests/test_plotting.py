#  Copyright (c) 2022 Robert Lieck.

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from musicflower.util import trisurf


class TestPlotting(TestCase):

    def test_trisurf(self):
        N = 10
        M = 50
        # generate points
        unique_points = np.sort(np.random.uniform(0, 1, (N, 3)), axis=0)
        # make duplicates
        points = np.concatenate([unique_points, unique_points], axis=0)
        # generate triangle indices
        indices = np.random.randint(0, 2 * N, (M, 3))
        # get the points from the indices
        triangles = points[indices]

        # feed to trisurf
        x, y, z, i, j, k, r = trisurf(triangles)
        # make sure the unique points were retrieved
        assert_array_equal(unique_points, np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1))
        # make sure the correct indices were returned
        new_indices = np.concatenate([i[:, None], j[:, None], k[:, None]], axis=1)
        self.assertFalse(np.array_equal(indices, new_indices))  # because some original indices refer to duplicates
        assert_array_equal(indices % N, new_indices)  # taking modulo maps to first (unique) occurrence
