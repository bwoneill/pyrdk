from rdkio.reader import *
import pyspark
from pyspark.mllib.clustering import PowerIterationClustering
from itertools import combinations
from scipy.spatial.distance import *

reader = FileReader('/mnt/shared/ss223/S223r4b1.dat')
combos = combinations(range(reader.size), 2)


def sim(i, j):
    reader.seek(i)
    d1 = reader.read()
    reader.seek(j)
    d2 = reader.read()
    return i, j, euclidean(d1.signal[0], d2.signal[0])


sc = pyspark.SparkContext()
rdd = sc.parallelize(combos)
rdd = rdd.map(sim)
pic = PowerIterationClustering().train(rdd, 5)
