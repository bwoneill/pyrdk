from rdkio.reader import *
import pyspark
from pyspark.mllib.clustering import PowerIterationClustering
from itertools import combinations
from scipy.spatial.distance import *
from sklearn.preprocessing import scale
from os import environ
from collections import Counter

# chekout dbscan

# reader = FileReader('%s/ss223/S223r10b1.dat' % environ.get('RDKDATA'))
reader = FileReader('rdkbwoneill/ss223/S223r10b1.dat', 's3')
combos = combinations(range(1000), 2)


def sim((i, j)):
    reader.seek(i)
    d1 = reader.read()
    d1.signal = scale(d1.signal, axis=1)
    reader.seek(j)
    d2 = reader.read()
    d2.signal = scale(d2.signal, axis=1)
    return i, j, euclidean(d1.signal[7], d2.signal[7])


sc = pyspark.SparkContext()
sc._conf.set('spark.executor.memory', '64g').set('spark.driver.memory', '64g').set('spark.driver.maxResultsSize', '0')
rdd = sc.parallelize(combos)
sim_rdd = rdd.map(sim)
# sim_rdd.cache()
# test = sim_rdd.collect()
pic = PowerIterationClustering().train(sim_rdd, 5)
labels = pic.assignments().collect()
count = Counter([a.cluster for a in labels])
print count
