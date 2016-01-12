import numpy as np
import pyspark
from itertools import izip
from operator import add

sc = pyspark.SparkContext()
rdd = sc.parallelize([])

eps = 1.
minRectSize = 2 * eps
dim = 3


def corner_count(_, vect):
    """
    Generate voxel lower corners for counting
    :param _: spark key, unused
    :param vect: tuple
    :return: tuple representation of lower corner, 1
    """
    return tuple(np.floor(np.array(vect) / minRectSize).astype(int)), 1


def min_seq_agg((v1, _), (v2, __)):
    return (min(x1, x2) for x1, x2 in izip(v1, v2))


def min_comb_agg(v1, v2):
    return (min(x1, x2) for x1, x2 in izip(v1, v2))


def max_seq_agg((v1, _), (v2, __)):
    return (max(x1, x2) for x1, x2 in izip(v1, v2))


def max_comb_agg(v1, v2):
    return (max(x1, x2) for x1, x2 in izip(v1, v2))


def in_voxel((v1, v2), (v3, v4)):
    return np.all(np.greater_equal(v2, v4)) and np.all(np.less_equal(v1, v3))


# create voxel (lowest corner) for each point and count points in each voxel
rectCorners_rdd = rdd.map(corner_count).aggregateByKey(0, add, add)
rectCorners_rdd.cache()
# grab one voxel
firstVect = rectCorners_rdd.first()[1]
# find the lower and upper bounds of the voxels
minCorner = rectCorners_rdd.aggregate(firstVect, min_seq_agg, min_comb_agg).collect()
maxCorner = rectCorners_rdd.aggregate(firstVect, max_seq_agg, max_comb_agg).collect()
boundingVoxel = (minCorner, maxCorner)
# find points in bounding voxel
toPartition = rectCorners_rdd.filter(lambda key, value: in_voxel(boundingVoxel, key)) \
    .aggregate(0, lambda total, (key, value): total + value, add).collect()
