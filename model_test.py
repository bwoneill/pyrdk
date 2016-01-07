from rdk.rdkio import *
from rdk.metrics import *
import pyspark
from pyspark.mllib.clustering import PowerIterationClustering, KMeans
from itertools import combinations, combinations_with_replacement, izip
from scipy.spatial.distance import *
from sklearn.preprocessing import scale
from os import environ
from collections import Counter
from math import exp
import numpy as np
import time

# chekout dbscan

# reader = FileReader('%s/ss223/S223r10b1.dat' % environ.get('RDKDATA'))
reader = FileReader('rdkbwoneill/ss223/S223r10b1.dat', 's3')
bucket = 'rdkbwoneill'
key = 'ss223/S223r10b1.dat'


def chunks(start, stop, chunk_size):
    result = []
    done = False
    index = start
    while not done:
        done = index + chunk_size >= stop
        if not done:
            result.append((index, chunk_size))
        else:
            result.append((index, min(chunk_size, stop - index)))
        index += chunk_size
    return result


def chunk_combinations(start, stop, chunk_size):
    return combinations_with_replacement(chunks(start, stop, chunk_size), 2)


def chunk_similarity(((i1, l1), (i2, l2))):
    result = []
    if i1 == i2:
        d1 = batchRead(bucket, key, i1, l1)
        for d in d1:
            d.ep_fit = fit_ep(d.signal[7])
            # d.signal = scale(d.signal, axis=1)
        d2 = d1
        combos = combinations(range(l1), 2)
    else:
        d1 = batchRead(bucket, key, i1, l1)
        for d in d1:
            d.ep_fit = fit_ep(d.signal[7])
            # d.signal = scale(d.signal, axis=1)
        d2 = batchRead(bucket, key, i2, l2)
        for d in d2:
            d.ep_fit = fit_ep(d.signal[7])
            # d.signal = scale(d.signal, axis=1)
        combos = [(i, j) for i in xrange(l1) for j in xrange(l2)]
    for i, j in combos:
        # similarity = exp(-euclidean(d1[i].signal[7], d2[j].signal[7]) ** 2)  # sweet spot between 2.35 and 2.36
        similarity = exp(-euclidean(d1[i].ep_fit, d2[i].ep_fit) ** 2)
        result.append((i1 + i, i1 + j, similarity))
    return result


def chunk_fits((i, l)):
    result = []
    data = batchRead(bucket, key, i, l)
    for j, d in enumerate(data):
        result.append((i + j, fit_ep(d.signal[7])))
    return result


def sim((i, j)):
    reader.seek(i)
    d1 = reader.read()
    d1.signal = scale(d1.signal, axis=1)
    reader.seek(j)
    d2 = reader.read()
    d2.signal = scale(d2.signal, axis=1)
    return i, j, exp(-euclidean(d1.signal[7], d2.signal[7]) ** 2)


if __name__ == '__main__':
    counts = []
    sc = pyspark.SparkContext()
    cchunks = list(chunk_combinations(0, 8192, 512))
    start = time.time()
    fit_rdd = sc.parallelize(chunks(0, 8182, 512)).flatMap(chunk_fits).sortByKey()
    fit_rdd.cache()
    values = fit_rdd.values()
    values.cache()
    km = KMeans().train(values, 5)
    predictions = km.predict(values)
    with open('predictions.txt', 'w') as f:
        f.write('index,p0,p1,p2,p3,category')
        for temp, pred in izip(fit_rdd.collect(), predictions.collect()):
            key, value = temp
            f.write('\n%i,%f,%f,%f,%f,%i' % (key, value[0], value[1], value[2], value[3], pred))
            pass
    print km.clusterCenters
    # rdd = sc.parallelize(cchunks)
    # sim_rdd = rdd.flatMap(lambda x: chunk_similarity(x))
    # pic = PowerIterationClustering().train(sim_rdd, 5)
    # labels = pic.assignments().collect()
    # count = Counter([a.cluster for a in labels])
    # print count
    print time.time() - start
    # with open('results.txt', 'w') as f:
    #     for a in labels:
    #         f.write('%i,%i\n' % (a.id, a.cluster))
