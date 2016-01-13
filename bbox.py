import numpy as np
import sys


class BoundingBox(object):
    """
    :lower: lower bounds of bounding box
    :upper: upper bounds of bounding box
    """
    def __init__(self, lower=None, upper=None, k=None):
        """
        :type lower: numpy.ndarray
        :param lower:
        :type upper: numpy.ndarray
        :param upper:
        :type k: int
        :param k: number of dimension
        """
        if lower is not None:
            self.lower = np.array(lower)
            self.upper = np.array(upper) if upper is not None else self.lower
        elif k is not None:
            self.lower = np.full(k, sys.float_info.max)
            self.upper = np.full(k, sys.float_info.min)
        else:
            self.lower = None
            self.upper = None

    def intersection(self, other):
        """
        :type other: BoundingBox
        :param other: BoundingBox to form intersection with
        :rtype: BoundingBox
        :return: intersection of bounding boxes
        """
        lower = np.maximum(self.lower, other.lower)
        upper = np.minimum(self.upper, other.upper)
        return BoundingBox(lower=lower, upper=upper)

    def union(self, other):
        """
        :type other: BoundingBox
        :param other: BoundingBox to form union with
        :rtype: BoundingBox
        :return: union of bounding boxes
        """
        lower = np.minimum(self.lower, other.lower)
        upper = np.maximum(self.upper, other.upper)
        return BoundingBox(lower=lower, upper=upper)

    def split(self, dim, value):
        """
        :type dim: int
        :param dim: dimension to split on
        :type value: float
        :param value: value to split on
        :rtype: BoundingBox, BoundingBox
        :return: result of split
        """
        left = BoundingBox(lower=np.copy(self.lower), upper=np.copy(self.upper))
        left.upper[dim] = value
        right = BoundingBox(lower=np.copy(self.lower), upper=np.copy(self.upper))
        right.lower[dim] = value
        return left, right

    def expand(self, eps=0, how='add'):
        """
        :type eps: float
        :param eps: expansion radius (may be a k-dim vector)
        :type how: str
        :param how: how to expand the box, must be 'add' or 'multiply'
        :rtype: BoundingBox
        :return: expanded bounding box
        """
        if how == 'add':
            return BoundingBox(self.lower - eps, self.upper + eps)
        elif how == 'multiply':
            span = self.upper - self.lower
            return BoundingBox(self.lower - eps * span, self.upper + eps * span)
        else:
            return None

    def contains(self, vector):
        """
        :type vector: numpy.ndarray
        :param vector: k-dim vector like
        :rtype: bool
        :return: True if vector is with the bounding box (lower bound is inclusive, upper is exclusive)
        """
        return np.all(self.lower <= vector) and np.all(self.upper >= vector)

    def __repr__(self):
        return 'BoundingBox(lower=%s\n\tupper=%s)' % (str(self.lower), str(self.upper))
