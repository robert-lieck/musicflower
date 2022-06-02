#  Copyright (c) 2022 Robert Lieck.

from typing import Union, Tuple
from itertools import repeat

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from triangularmap import TMap


def get_fourier_component(pcds: np.ndarray, fourier_component: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the amplitude and phase of a specific Fourier component.

    :param pcds: array of arbitrary shape, the Fourier transform is computed along the last dimension
    :param fourier_component: component of the Fourier transform to extract (must not be larger than the last dimension
     of `pcds`)
    :return: amplitude, phase (arrays of same shape as `pcds` without the last dimension)
    """
    phase = np.angle(np.fft.rfft(pcds, axis=-1))[..., fourier_component]
    amplitude = np.abs(np.fft.rfft(pcds, axis=-1))[..., fourier_component]
    return amplitude, phase


def start_duration(n):
    start = []
    duration = []
    for idx in range(1, n + 1):
        start.append(np.arange(idx) / n)
        duration.append(np.ones(idx) - idx / n)
    return np.concatenate(start), np.concatenate(duration)


def remap_to_xyz(amplitude: np.ndarray, phase: np.ndarray, inner_radius: float = 0.4, inverted: bool = False,
                 spherical: bool = True, rescale_func: callable = np.sqrt, axis: int = -1
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    :return: x, y, z: 1D arrays of length :math:`k` containing the Cartesian coordinates.
    """
    # check shapes
    assert amplitude.shape == phase.shape, f"'amplitude' and 'phase' have to have the same shape but have " \
                                           f"{amplitude.shape} and {phase.shape}, respectively."
    # rescale
    if rescale_func is not None:
        amplitude = rescale_func(amplitude)
    # get the durations
    _, duration = start_duration(TMap.n_from_size1d(phase.shape[axis]))
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
    else:
        z = amplitude
    return x, y, z


def get_time_traces(x: np.ndarray, y: np.ndarray, z: np.ndarray, colors: np.ndarray, n_steps: int,
                    axis=0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute `n_steps` + 1 traces that run from the top of the triangular map to its bottom by interpolating along rows. The
    input arrays can have an arbitrary number of additional batch dimensions.

    Each trace has the same number of points as there are rows in the triangular map. As time runs from 0 to 1, the
    points run from left to right along the rows, interpolating linearly between points.

    :param x: array with x-coordinates
    :param y: array with y-coordinates
    :param z: array with z-coordinates
    :param colors: array with RGB colours (last dimension must correspond to colours and have size 3)
    :param n_steps: create `n_steps` + 1 traces from time 0 to time 1
    :param axis: Axis of the input arrays that corresponds to the triangular map. It must have a compatible length (i.e.
     a length of :math:`n(n+1)/2` for some integer :math:`n`).
    :return: xyz, colors: two arrays of shape (n_steps + 1, n, ..., 3) with xyz-coordinates and colours, where `...`
     corresponds to any additional batch dimensions.
    """
    assert x.shape == y.shape == z.shape == colors.shape[:-1]
    assert colors.shape[-1] == 3
    assert len(colors.shape) >= 2
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
                arr_out[:, depth, ..., :] = arr.dslice(depth)[0, ..., :]
            else:
                arr_out[:, depth, ..., :] = interpolate.interp1d(np.linspace(0, 1, depth + 1),
                                                              arr.dslice(depth)[..., :],
                                                              axis=0,
                                                              copy=False)(np.linspace(0, 1, n_steps + 1))
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
        n = TMap.n_from_size1d(n.shape[axis])
    # get TMap of point indices
    index_map = TMap(np.arange(TMap.size1d_from_n(n)))
    # iterate through rows and construct triangles
    i, j, k = [], [], []
    for d in range(n - 1):
        a_slice = index_map.dslice(d)
        b_slice = index_map.dslice(d + 1)
        # triangles with tip up
        i.append(a_slice)
        j.append(b_slice[1:])
        k.append(b_slice[:-1])
        # triangles with tip down
        if d < n - 2:
            c_slice = index_map.dslice(d + 2)
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


def iterable_or_repeat(it, exclude=()):
    if isinstance(it, exclude):
        return it
    try:
        iter(it)
        return it
    except TypeError:
        return repeat(it)


def broadcast_func(func, **kwargs):
    ret = []
    for vals in zip(*list(kwargs.values())):
        ret.append(func(**dict(zip(kwargs.keys(), vals))))
    return tuple(ret)


def bezier(t, p):
    if len(p.shape) > 1:
        t = t[:, None]
    if p.shape[0] == 2:
        return (1 - t) * p[0] + t * p[1]
    elif p.shape[0] == 3:
        return (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2])
    elif p.shape[0] == 4:
        return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
    else:
        raise NotImplementedError("Bezier curve only implemented for 2–4 points, i.e., 0–2 control points "
                                  "(linear, quadratic, cubic)")


def show_bezier(p):
    plt.plot(*bezier(np.linspace(0, 1, 300), p).T, '-')
    plt.plot(p[:, 0], p[:, 1], 'o-')
    plt.show()


def main():
    show_bezier(np.array([[0, 0],
                          [0, 1],
                          [1, 1]]))

    # t = np.linspace(0, 1, 300)
    # p = np.array([0, 10, 0.9, 1])
    # plt.plot(t, bezier(t, p).T, '-')
    # # plt.plot(p[:, 0], p[:, 1], 'o-')
    # plt.show()


if __name__ == "__main__":
    main()
