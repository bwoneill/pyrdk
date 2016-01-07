import matplotlib.pyplot as plt
from rdk.rdkio import *
import numpy as np
from sklearn.preprocessing import scale

reader = FileReader('/mnt/shared/ss223/S223r10b1.dat')
clusters = np.loadtxt('results.txt', delimiter=',').astype(int)

x = np.arange(512, 1024)
count = 0

for index, cluster in clusters:
    if count < 20 or cluster != 0:
        if cluster == 0:
            count += 1
        reader.seek(index)
        data = reader.read()
        data.signal = scale(data.signal, axis=1)
        y = data.signal[7][512:1024]
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        plt.xlim(512, 1024)
        plt.savefig('plots/c%i/S223r10b1e%i.png' % (cluster, index))
        plt.close()
