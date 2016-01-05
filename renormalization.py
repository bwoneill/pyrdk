from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from rdkio.reader import *
import numpy as np
from copy import copy
from multiprocessing import Pool

f = FileReader('/mnt/shared/ss223/S223r4b1.dat')

n = 0
m1 = np.zeros((8, 2048))
m2 = np.zeros((8, 2048))

while f.tell() < f.size:
    data = f.read()
    # data.signal = scale(data.signal, axis=1)
    m1 += data.signal
    m2 += data.signal ** 2
    n += 1
    if n % 1000 == 0:
        print n

m1 /= n
m2 /= n
std = m2 - m1 ** 2

f.seek(0)


def rescale(data):
    d1 = copy(data)
    d2 = copy(data)
    d1.signal = (d1.signal - m1) / std
    d1.signal = scale(d1.signal, axis=1)
    d2.signal = scale(d2.signal, axis=1)
    return d1, d2


def plot(data):
    d1, d2 = rescale(data)
    _, ax = plt.subplots(3, 3, sharex=True)
    x = range(2048)
    for i in xrange(8):
        temp = ax[i / 3][i % 3]
        temp.plot(x, d1.signal[i])
        temp.plot(x, d2.signal[i])
        temp.set_xlim(0, 2047)
    plt.show()
