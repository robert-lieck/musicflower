#  Copyright (c) 2022 Robert Lieck.

import numpy as np
import matplotlib.pyplot as plt

from TriangularMap import TMap


def get_fourier_component(pcds, fourier_component, rescale_func=np.sqrt):
    angle = np.angle(np.fft.rfft(pcds, axis=-1))[:, fourier_component]
    amplitude = np.abs(np.fft.rfft(pcds, axis=-1))[:, fourier_component]
    if rescale_func is not None:
        amplitude = rescale_func(amplitude)
    return amplitude, angle


def start_duration(n):
    start = []
    duration = []
    for idx in range(1, n + 1):
        start.append(np.arange(idx) / n)
        duration.append(np.ones(idx) - idx / n)
    return np.concatenate(start), np.concatenate(duration)


def remap_to_xyz(amplitude, angle, inner_radius, inverted, spherical):
    start, duration = start_duration(TMap.n_from_size1d(angle.shape[0]))
    if inverted:
        radius = (1 - duration) + inner_radius
    else:
        radius = duration + inner_radius
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    if spherical:
        theta = (1 - amplitude) * np.pi / 2
        x *= np.cos(theta)
        y *= np.cos(theta)
        z = radius * np.sin(theta)
    else:
        z = amplitude
    return x, y, z


def bezier(t, p):
    if len(p.shape) > 1:
        t = t[:, None]
    match p.shape[0]:
        case 2:
            return (1 - t) * p[0] + t * p[1]
        case 3:
            return (1 - t) * ((1 - t) * p[0] + t * p[1]) + t * ((1 - t) * p[1] + t * p[2])
        case 4:
            return (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t ** 2 * p[2] + t ** 3 * p[3]
        case _:
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
