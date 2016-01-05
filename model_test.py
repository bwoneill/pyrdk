from rdkio.reader import *
import pyspark
from pyspark.mllib.clustering import PowerIterationClustering
from itertools import combinations
from scipy.spatial.distance import *
from os import environ

# reader = FileReader('%s/ss223/S223r10b1.dat' % environ.get('RDKDATA'))
reader = FileReader('rdkbwoneill/ss223/S223r10b1.dat', False)
combos = combinations(range(reader.size), 2)


def sim(i, j):
    reader.seek(i)
    d1 = reader.read()
    reader.seek(j)
    d2 = reader.read()
    return i, j, euclidean(d1.signal[0], d2.signal[0])


# sc = pyspark.SparkContext()
# sc._conf.set('spark.executor.memory', '64g').set('spark.driver.memory', '64g').set('spark.driver.maxResultsSize', '0')
# rdd = sc.parallelize(combos)
# rdd = rdd.map(sim)
# pic = PowerIterationClustering().train(rdd, 5)
