from kdpart import KDPartitioner
import pyspark as ps
from scipy.spatial.distance import *
import sklearn.cluster as skc


def dbscan_partition(iterable, params):
    """
    :type iterable: iter
    :param iterable: iterator yielding ((key, partition), vector)
    :type params: dict
    :param params: dictionary containing sklearn DBSCAN parameters
    :rtype: iter
    :return: ((key, cluster_id), v)
    Performs a DBSCAN on a given partition of the data
    """
    # read iterable into local memory
    data = list(iterable)
    (key, part), vector = data[0]
    x = np.array([v for (_, __), v in data])
    y = np.array([k for (k, _), __ in data])
    # perform DBSCAN
    model = skc.DBSCAN(**params)
    c = model.fit_predict(x)
    # yield ((key, cluster_id), v)
    for i in xrange(len(c)):
        yield ((y[i], '%i:%i' % (part, c[i])), x[i])


def reduce_cluster_id(((key, cluster_id), v), cluster_dict):
    cluster_id = cluster_id.split(',')[0]
    if '-1' not in cluster_id and cluster_id in cluster_dict:
        return (key, cluster_dict[cluster_id]), v
    else:
        return (key, -1), v


def filter((key, value), boxes):
    pass


class DBSCAN(object):
    def __init__(self, eps=0.5, min_samples=5, metric=euclidean, max_partitions=None):
        """
        :type eps: float
        :param eps: nearest neighbor radius
        :type min_samples: int
        :param min_samples: minimum number of samples with radius eps
        :type metric: callable
        :param metric: distance metric (should be scipy.spatial.distance.euclidian or scipy.spatial.distance.cityblock)
        :type max_partitions: int
        :param max_partitions: maximum number of partitions in KDPartitioner
        Using a metric other than euclidian or cityblock/Manhattan may not work as the bounding boxes expand in
        such a way that other metrics may return distances less than eps for points outside the box.
        """
        self.eps = eps
        self.min_samples = int(min_samples)
        self.metric = metric
        self.max_partitions = max_partitions
        self.data = None
        self.result = None

    def train(self, data):
        """
        :type data: pyspark.RDD
        :param data: (key, k-dim vector like)
        """
        parts = KDPartitioner(data, self.max_partitions)
        # self.data = parts.result
        self.data = data
        neighbors = {}
        self.bounding_boxes = parts.bounding_boxes
        self.expanded_boxes = {}
        # expand bounding boxes to include neighbors within eps
        new_data = sc.emptyRDD()
        for label, box in parts.bounding_boxes.iteritems():
            expanded_box = box.expand(1.5 * self.eps)
            self.expanded_boxes[label] = expanded_box
            neighbors[label] = self.data.filter(lambda (k, v): expanded_box.contains(v)) \
                .map(lambda (k, v): ((k, label), v))
            new_data = new_data.union(neighbors[label])
        # merge those labeled neighbors into the data set
        self.neighbors = neighbors
        # self.data = sc.emptyRDD()
        # for label, rdd in neighbors.iteritems():
        #     self.data = self.data.union(rdd)
        self.data = new_data
        # repartition data set on the partition label
        self.data = self.data.map(lambda ((k, p), v): (p, (k, v))) \
            .partitionBy(len(parts.partitions)) \
            .map(lambda (p, (k, v)): ((k, p), v))
        params = {'eps': self.eps, 'min_samples': self.min_samples, 'metric': self.metric}
        # perform dbscan on each part
        self.data = self.data.mapPartitions(lambda iterable: dbscan_partition(iterable, params))
        self.data.cache()
        # merge resulting clusters
        point_labels = self.data.map(lambda ((k, c), v): (k, c)).groupByKey() \
            .map(lambda (k, c): (k, list(c))).collect()
        # make cluster dict to map local clusters into global clusters
        new_cluster_label = 0
        cluster_dict = {}
        # with open('test.log', 'w') as f:
        #     f.write('key,clusters')
        #     for key, cluster_ids in point_labels:
        #         f.write('\n%i,%s' % (key, ';'.join(np.array(list(cluster_ids)).astype(str))))
        for k, cluster_ids in point_labels:
            cluster_ids = np.array(list(cluster_ids))
            in_dict = np.array([cluster_id in cluster_dict for cluster_id in cluster_ids])
            if np.any(in_dict):
                # find lowest label for labeled clusters
                labeled_clusters = cluster_ids[in_dict]
                labels = [cluster_dict[cluster_id] if cluster_id in cluster_dict else new_cluster_label
                          for cluster_id in cluster_ids]
                # labels = [cluster_dict[cluster_id] for cluster_id in labeled_clusters]
                label = np.min(labels)
                for key, value in cluster_dict.iteritems():
                    if value in labels and value != label:
                        cluster_dict[key] = label
            else:
                # create/increment label
                label = new_cluster_label
                new_cluster_label += 1
            for cluster_id in cluster_ids:
                # for each cluster_id
                if '-1' not in cluster_id:
                    cluster_dict[cluster_id] = label
        self.cluster_dict = cluster_dict
        self.result = self.data.map(lambda x: reduce_cluster_id(x, cluster_dict)) \
            .map(lambda ((k, c), v): (k, c)).reduceByKey(min).sortByKey()
        self.result.cache()

    def assignments(self):
        """
        :rtype: pyspark.RDD
        :return: (key, cluster_id)
        Retrieve the results of the DBSCAN
        """
        return self.result.collect()


if __name__ == '__main__':
    # Example of pypadis.DBSCAN
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    from time import time
    from itertools import izip

    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    sc = ps.SparkContext()
    test_data = sc.parallelize(enumerate(X))
    start = time()
    dbscan = DBSCAN(0.3, 10)
    dbscan.train(test_data)
    result = np.array(dbscan.assignments())
    print time() - start
    clusters = result[:, 1]
    # temp = dbscan.data.glom().collect()
    # for t in temp:
    #     x = [t2[1][0] for t2 in t]
    #     y = [t2[1][1] for t2 in t]
    #     c = [int(t2[0][1].split(':')[1]) for t2 in t]
    #     l = [int(t2[0][1].split(':')[0]) for t2 in t]
    #     box = dbscan.expanded_boxes[l[0]]
    #     in_box = [box.contains([a, b]) for a, b in izip(x, y)]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     ax.add_patch(patches.Rectangle(box.lower, *(box.upper - box.lower), alpha=0.2))
    #     print in_box
    #     plt.scatter(x, y, c=c)
    #     plt.show()
    x = X[:, 0]
    y = X[:, 1]
    plt.set_cmap('jet')
    plt.scatter(x, y, c=clusters)
    plt.show()
