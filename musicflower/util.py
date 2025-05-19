#  Copyright (c) 2022 Robert Lieck.

from typing import Union, Tuple
from itertools import repeat
import math

import numpy as np
from scipy import interpolate

from pitchscapes.keyfinding import KeyEstimator
from triangularmap import TMap


rad_to_deg = 360 / (2 * math.pi)


def feature_exists(app, name):
    return name in app.feature_extractors or name in app.feature_remappers


def get_fourier_component(pcds: np.ndarray, fourier_component: int = None) -> np.ndarray:
    """
    Compute the amplitude and phase of one or all Fourier components of a real-valued input.

    :param pcds: array of arbitrary shape `(..., n)` with pitch-class distributions of size n along the last dimension,
     along which the Fourier transform is computed using `numpy.fft.rfft`.
    :param fourier_component: None (default) to return all components or int specifying the component to return (must
     not be larger than ` n//2`)
    :return:  array of shape (2, ..., n//2 + 1) with amplitude and phase along the first dimension and the different
     Fourier components along the last dimension
    """
    pcds = np.asarray(pcds, dtype=float)
    pcds /= pcds.sum(axis=-1, keepdims=True)
    ft = np.fft.rfft(pcds, axis=-1)
    if fourier_component is not None:
        ft = ft[..., fourier_component]
    phase = np.angle(ft)
    amplitude = np.abs(ft)
    return np.concatenate([amplitude[None], phase[None]])


def start_duration(n):
    start = []
    duration = []
    for idx in range(1, n + 1):
        start.append(np.arange(idx) / n)
        duration.append(np.ones(idx) - (idx - 1) / n)
    return np.concatenate(start), np.concatenate(duration)


def remap_to_xyz(amplitude: np.ndarray, phase: np.ndarray, inner_radius: float = 0.2, inverted: bool = False,
                 spherical: bool = True, rescale_func: callable = np.sqrt, axis: int = -1,
                 theta_r=False, scape2D=False,
                 ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Map a triangular map with amplitudes and phases to 3D space. The `axis` representing the triangular map must have a
    size of :math:`k=n(n+1)/2` for some integer :math:`n` to represent a valid triangular map (:math:`n` is the
    resolution of the triangular map). The default is `axis=-1`, that is, the last axis.

    :param amplitude: array with amplitudes in the interval :math:`[0, 1]`
    :param phase: array with phases in radians
    :param inner_radius: an offset to avoid collapsing points at the center
    :param inverted: invert the radial dimension
    :param spherical: use spherical coordinates to map to the upper half-sphere (zero amplitudes map to the zenith/top;
     amplitudes of one map to the horizontal plane); if `False`, cylindrical coordinates are used (the amplitude is used
     as the vertical dimension)
    :param rescale_func: function to rescale amplitudes; this should monotonically rescale the interval :math:`[0, 1]`;
     the default uses np.sqrt, which distorts towards high amplitudes, i.e., it "opens up the top of the flower".
    :param axis: axis of the input arrays that represents the triangular map
    :param theta_r: also return theta an r if spherical=True
    :param scape2D: map to a conventional 2D scape plot in the x-y-plane instead
    :return: x, y, z: 1D arrays of length :math:`k` containing the Cartesian coordinates.
    """
    # check shapes
    assert amplitude.shape == phase.shape, f"'amplitude' and 'phase' have to have the same shape but have " \
                                           f"{amplitude.shape} and {phase.shape}, respectively."
    # rescale
    if rescale_func is not None:
        amplitude = rescale_func(amplitude)
    # get start and duration of points
    start, duration = start_duration(TMap.n_from_size(phase.shape[axis]))
    # 2D or 3D
    if scape2D:
        x = start + duration / 2
        y = duration
        z = np.zeros_like(x)
        return x, y, z
    else:
        # convert to radial component
        if inverted:
            radius = (1 - duration) + inner_radius
        else:
            radius = duration + inner_radius
        # adapt shape for broadcasting
        shape = np.ones(len(phase.shape), dtype=int)
        shape[axis] = len(radius)
        radius = radius.reshape(shape)
        # compute Cartesian coordinates
        x = np.cos(phase) * radius
        y = np.sin(phase) * radius
        if spherical:
            # theta is the angle from the horizontal plane to the zenith
            # --> map amplitude in [0, 1] to theta in [pi/2, 0]
            # i.e. an amplitude of zero corresponds to theta=pi/2 (zenith)
            # an amplitude of one corresponds to theta=0 (horizontal plane)
            theta = (1 - amplitude) * np.pi / 2
            x *= np.cos(theta)
            y *= np.cos(theta)
            z = radius * np.sin(theta)
            if theta_r:
                return x, y, z, theta, radius
            else:
                return x, y, z
        else:
            z = amplitude
            if theta_r:
                raise ValueError("Can only return theta and r for spherical=True")
            return x, y, z


def get_time_traces(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, n_steps: int = None,
                    times: np.ndarray = None, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute `n_steps` + 1 traces that run from the top of the triangular map to its bottom by interpolating along
    rows. The input arrays can have an arbitrary number of additional batch dimensions.

    Each trace has the same number of points as there are rows in the triangular map. As time runs from 0 to 1, the
    points run from left to right along the rows, interpolating linearly between points.

    :param x: array with x-coordinates
    :param y: array with y-coordinates
    :param z: array with z-coordinates
    :param colors: array with RGB colours (last dimension must correspond to colours and have size 3)
    :param n_steps: create `n_steps` + 1 traces from time 0 to time 1 (instead of using explicit ``times``)
    :param times: array with values in [0, 1]; create traces for these times (instead of ``n_steps`` equally spaced)
    :param axis: Axis of the input arrays that corresponds to the triangular map. It must have a compatible length (i.e.
     a length of :math:`n(n+1)/2` for some integer :math:`n`).
    :return: xyz, colors: two arrays of shape (n_steps + 1, n, ..., 3) with xyz-coordinates and colours, where `...`
     corresponds to any additional batch dimensions.
    """
    assert x.shape == y.shape == z.shape == colors.shape[:-1]
    assert colors.shape[-1] == 3
    assert len(colors.shape) >= 2
    assert (n_steps is None) != (times is None); "exactly one of 'n_steps' or 'times' has to be provided"
    if n_steps is None:
        assert len(times) == 1
        n_steps = len(times) - 1
    if times is None:
        times = np.linspace(0, 1, n_steps + 1)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    if axis != 0:
        xyz = np.moveaxis(xyz, axis, 0)
        colors = np.moveaxis(colors, axis, 0)
    batch_shape = xyz.shape[1:-1]
    xyz = TMap(xyz)
    colors = TMap(colors)
    n = colors.n
    xyz_out = np.zeros((n_steps + 1, n) + batch_shape + (3,))
    colors_out = np.ones((n_steps + 1, n) + batch_shape + (3,))
    for depth in range(n):
        for arr, arr_out in [(xyz, xyz_out), (colors, colors_out)]:
            # for i in range(3):
            if depth == 0:
                arr_out[:, depth, ..., :] = arr.dslice[depth][0, ..., :]
            else:
                arr_out[:, depth, ..., :] = interpolate.interp1d(np.linspace(0, 1, depth + 1),
                                                                 arr.dslice[depth][..., :],
                                                                 axis=0,
                                                                 copy=False)(times)
    return xyz_out, colors_out


def trisurf(triangles, decimals=None):
    """
    Convert a Nx3x3 array of N triangles in 3D space, defined via their corners, into x, y, z coordinate arrays, and
    i, j, k index arrays as used by Plotly's Mesh3d function. Additionally, a return index r is return that can be used
    to reconstruct information associated  with the original points.

    :param triangles: Nx3x3 array of triangle vertices in 3D space (last dimension is x-y-z coordinate)
    :param decimals: decimals to use for rounding using numpy.around to improve point identification
     (default: None i.e. do not round)
    :return: x, y, z, i, j, k, r
    """
    # asser shape is OK
    assert triangles.shape[1:] == (3, 3), f"'triangles' has to have shape Nx3x3 but has shape {triangles.shape}"
    # round if requested
    if decimals is not None:
        triangles = np.around(triangles, decimals=decimals)
    # flatten to array of 3D points
    triangle_points = triangles.reshape(-1, 3)
    # get unique points (this also sorts!)
    unique_points, return_index = np.unique(triangle_points, axis=0, return_index=True)
    # convert to array of tuples for index finding
    triangle_points_as_tuples = np.array([tuple(x) for x in triangle_points], dtype="f,f,f")
    unique_points_as_tuples = np.array([tuple(x) for x in unique_points], dtype="f,f,f")
    # find indices of triangle points (relies on sorted unique points)
    triangle_point_indices = np.searchsorted(unique_points_as_tuples, triangle_points_as_tuples)
    # reshape to array of 3D indices
    triangle_point_indices = triangle_point_indices.reshape(-1, 3)
    # return separate arrays
    return (unique_points[:, 0], unique_points[:, 1], unique_points[:, 2],
            triangle_point_indices[:, 0], triangle_point_indices[:, 1], triangle_point_indices[:, 2],
            return_index)


def surface_scape_indices(n: Union[int, np.ndarray], axis: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return `i`, `j`, `k` point indices for triangles spanned between points of a triangular map.

    A triangular map of resolution :math:`n=4` has :math:`n(n+1)/2=10` points (left), which define
    :math:`(n-1)^2=9` triangles (right):

    ::

             .                      .
            /0\\                    /0\\
           .___.                  .___.
          /1\\ /2\\                /1\\2/3\\
         .___.___.              .___.___.
        /3\\ /4\\ /5\\            /4\\5/6\\7/8\\
       .___.___.___.          .___.___.___.
       6   7   8   9

    The returned point indices `i`, `j`, `k` define the vertices of these triangles, so for the first/top triangle, we
    have `i[0]=0`, `j[0]=1`, `k[0]=2`, for the second `i[1]=1`, `j[1]=3`, `k[1]=4` etc. The order of the triangles and
    their normal (defined by the order of the vertices) may be different from this example, but the normals are
    guaranteed to point in the same direction for all triangles. It is assumed that points are given in the top-down
    order from this example, which is the convention used throughout this package and by the :class:`TMap` class.

    :param n: resolution of the triangular map or an array with a triangular map (`axis` specified which dimension
     corresponds to the triangular map)
    :param axis: if `n` is an array, `axis` has to be provided to specify the dimension corresponding to the triangular
     map
    :return: i, j, k
    """
    # get size from array
    if isinstance(n, np.ndarray):
        if axis is None:
            raise TypeError(f"If an array is provided for 'n', 'axis' has to be specified, too.")
        n = TMap.n_from_size(n.shape[axis])
    # get TMap of point indices
    index_map = TMap(np.arange(TMap.size_from_n(n)))
    # iterate through rows and construct triangles
    i, j, k = [], [], []
    for d in range(n - 1):
        a_slice = index_map.dslice[d]
        b_slice = index_map.dslice[d + 1]
        # triangles with tip up
        i.append(a_slice)
        j.append(b_slice[1:])
        k.append(b_slice[:-1])
        # triangles with tip down
        if d < n - 2:
            c_slice = index_map.dslice[d + 2]
            i.append(b_slice[:-1])
            j.append(b_slice[1:])
            k.append(c_slice[1:-1])
    i, j, k = np.concatenate(i), np.concatenate(j), np.concatenate(k)
    assert i.shape == j.shape == k.shape and len(i) == (n - 1) ** 2, \
        f"i, j, k should have length {(n - 1) ** 2} but have shape {i.shape}, {j.shape}, {k.shape}. " \
        f"This is a bug in the code."
    return i, j, k


def assert_valid_corpus(corpus):
    if not 2 <= len(corpus.shape) <= 3:
        raise ValueError(f"A corpus should be an array with 2 or 3 dimension, corresponding to a single or multiple "
                         f"pieces, but the provided corpus has {len(corpus.shape)}")


def assert_valid_xyz_col(x, y, z, colors):
    if not (x.shape == y.shape == z.shape == colors.shape[:-1]) or colors.shape[-1] not in (3, 4):
        raise ValueError(f"x, y, z must have the same shape; the first dimensions of colors must be the same as the "
                         f"shape of x, y, and z; the last dimension of colors must be 3 or 4. We got (x/y/z/colors): "
                         f"{x.shape}/{y.shape}/{z.shape}/{colors.shape}")


def transpose_profiles(profile, modulo=False):
    """
    Returns two arrays representing the circle of fifth profiles.

    :param profile: array of arbitrary shape with profiles along first dimension
    :param modulo: change from chromatic to fifth-based order or vice versa
    :return: transposition of the key profile, reordered in the circle of fifth order.

    """
    dimension = profile.shape[0]
    # computing all possible transpositions
    transposed = np.concatenate([np.roll(profile, shift=roll_idx, axis=0)[None] for roll_idx in range(dimension)])
    if modulo:
        return tonal_modulo(transposed, axis=-2)
    else:
        return transposed


def tonal_modulo(arr: np.array, axis=-1):
    s = [slice(None)] * len(arr.shape)
    s[axis] = (np.arange(12) * 7) % 12
    return arr[tuple(s)]


def iterable_or_repeat(it, exclude=()):
    if isinstance(it, exclude):
        return it
    try:
        iter(it)
        return it
    except TypeError:
        return repeat(it)


def repeat_kwargs(kwargs):
    return {key: repeat(val) for key, val in kwargs.items()}


def broadcast_func(func, **kwargs):
    ret = []
    for vals in zip(*list(kwargs.values())):
        ret.append(func(**dict(zip(kwargs.keys(), vals))))
    return tuple(ret)


major_minor_profiles = KeyEstimator.profiles['albrecht']
major_minor_profiles = np.array([major_minor_profiles['major'], major_minor_profiles['minor']]).T  # (pitch, mode)
major_minor_profiles = transpose_profiles(major_minor_profiles)  # (trans, pitch, mode)
major_minor_profiles = np.moveaxis(major_minor_profiles, 2, 0)  # (mode, trans, pitch)
