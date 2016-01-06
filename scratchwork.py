import numpy as np
from itertools import combinations, combinations_with_replacement


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
