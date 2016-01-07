from scipy.spatial.distance import *
from scipy.optimize import *
from sklearn.preprocessing import scale
from rdkio import *
import numpy as np


def optimized_distance(v1, v2, dist=euclidean):
    params = [1., 0., 0.]
    # v1 = scale(v1)
    # v2 = scale(v2)
    f = lambda p: ss_distance(v1, v2, p[0], p[1], p[2], dist=dist)
    params = minimize(f, params).x
    return f(params), params


def ss_distance(v1, v2, s, shift_up, shift_right, dist=euclidean):
    return dist(np.roll(v1, int(shift_right)) * s + shift_up, v2 / s - shift_up)


def ep_scale(v):
    return scale(v)


def ep_scale_shift(v):
    peak = np.argmax(v)
    v_temp = np.roll(v, 1023 - peak)  # move peak to center of signal
    return scale(v_temp[959: 1215])  # use only the 64 points before and 192 points after peak
    # (256 points or 1/8th of total signal)


def ep_dist(v1, v2):
    return euclidean(ep_scale(v1), ep_scale(v2))


def e_func(x, a, b, c, d):
    return a + b * np.exp(-(x - c) ** 2 / d)


std_x = np.arange(2048)


def fit_ep(v):
    params = [0, np.max(v), np.argmax(v), 100]
    try:
        params = curve_fit(e_func, std_x, v, params)[0].tolist()
    except RuntimeError as e:
        params = [0, 0, 0, 1]
    res = np.sum((v - e_func(std_x, *params)) ** 2)
    params.append(res)
    return params


if __name__ == '__main__':
    reader = FileReader('/mnt/shared/ss223/S223r10b1.dat')
    d1 = reader.read()
    for _ in xrange(20):
        d2 = reader.read()
        # print optimized_distance(d1.signal[7], d2.signal[7], euclidean)
        # print euclidean(ep_scale_shift(d1.signal[7]), ep_scale_shift(d2.signal[7]))
        print fit_ep(d2.signal[7])
