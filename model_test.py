from rdkio.reader import *
import pyspark
from pyspark.mllib.clustering import PowerIterationClustering
from itertools import combinations, combinations_with_replacement
from scipy.spatial.distance import *
from sklearn.preprocessing import scale
from os import environ
from collections import Counter
from math import exp
import numpy as np

# chekout dbscan

# reader = FileReader('%s/ss223/S223r10b1.dat' % environ.get('RDKDATA'))
reader = FileReader('rdkbwoneill/ss223/S223r10b1.dat', 's3')
bucket = 'rdkbwoneill'
key = 'ss223/S223r10b1.dat'


# combos = combinations(range(1000), 2)

def make_chunks(start, stop, chunk_size):
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
    return combinations_with_replacement(result, 2)


def chunk_similarity(((i1, l1), (i2, l2))):
    result = []
    if i1 == i2:
        d1 = batchRead(bucket, key, i1, l1)
        d2 = d1
        combos = combinations(range(l1), 2)
    else:
        d1 = batchRead(bucket, key, i1, l1)
        d2 = batchRead(bucket, key, i2, l2)
        combos = [(i, j) for i in xrange(l1) for j in xrange(l2)]
    for i, j in combos:
        if d1[i].start_flat != '********':
            print i1 + i
        similarity = exp(-euclidean(d1[i].signal[7], d2[j].signal[7]) ** 2)
        result.append((i1 + i, i1 + j, similarity))
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
    sc = pyspark.SparkContext()
    sc._conf.set('spark.executor.memory', '64g').set('spark.driver.memory', '64g').set('spark.driver.maxResultsSize',
                                                                                       '0')
    chunks = list(make_chunks(0, 1024, 32))
    rdd = sc.parallelize(chunks)
    sim_rdd = rdd.flatMap(chunk_similarity)
    # rdd = sc.parallelize(combos)
    # sim_rdd = rdd.map(sim)
    sim_rdd.cache()
    test = sim_rdd.collect()
    # pic = PowerIterationClustering().train(sim_rdd, 2)
    # labels = pic.assignments().collect()
    # count = Counter([a.cluster for a in labels])
    # print count
