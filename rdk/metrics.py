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


def ep_scale_shift(v):
    peak = np.argmax(v)
    return scale(v[peak - 64: peak + 192])


def ep_dist(v1, v2):
    return euclidean(ep_scale_shift(v1), ep_scale_shift(v2))


if __name__ == '__main__':
    reader = FileReader('/mnt/shared/ss223/S223r10b1.dat')
    d1 = reader.read()
    for _ in xrange(20):
        d2 = reader.read()
        # print optimized_distance(d1.signal[7], d2.signal[7], euclidean)
        print euclidean(ep_scale_shift(d1.signal[7]), ep_scale_shift(d2.signal[7]))