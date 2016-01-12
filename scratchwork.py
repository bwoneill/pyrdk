import numpy as np
from itertools import combinations, combinations_with_replacement
from pyPDSDBSCAN.voxel import Voxel


def distributed_combinations(l, n):
    size = int(np.ceil(float(len(l)) / n))
    splits = size * np.arange(1, n)
    lists = np.split(l, splits)
    combos = combinations_with_replacement(lists, 2)
    result = []
    for l1, l2 in combos:
        if l1[0] == l2[0]:
            result += list(combinations(l1, 2))
        else:
            result += [(x1, x2) for x1 in l1 for x2 in l2]
    return result


if __name__ == '__main__':
    import pyspark
    from operator import add

    p1 = np.array((0, 0, 0))
    p2 = np.array((1, 1, 1))
    p3 = np.array((0, 1, 0))
    p4 = np.array((1, 2, 1))
    p5 = np.array([0, 0, 0])
    p6 = np.array([1, 1, 1])
    v1 = Voxel(p1, p2)
    v2 = Voxel(p3, p4)
    v3 = Voxel(p5, p6)
    sc = pyspark.SparkContext()
    rdd = sc.parallelize([(((0, 0), (1, 1)), 1), (((0, 0), (1, 1)), 1), (((0, 1), (1, 2)), 1)])
    print rdd.aggregateByKey(0, add, add).collect()
