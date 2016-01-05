from rdkio.reader import *
from sklearn.preprocessing import scale
import pyspark as ps
import boto3
import json

# need to copy credentials to all machines in cluster

with open('/home/bwoneill/.aws/rootkey.json') as f:
    keys_dict = json.load(f)
keys = (keys_dict['AWSAccessKeyId'], keys_dict['AWSSecretKey'])


def dat_to_events(file_name):
    result = []
    reader = FileReader(file_name, False)
    data = reader.read()
    while data is not None:
        data.signal = scale(data.signal, axis=1)
        for signal in data.signal:
            result.append(signal)
        data = reader.read()
    return result


sc = ps.SparkContext()

# list of files in series
list_of_files = {'rdkbwoneill/ss100/S223r10b1.dat',
                 'rdkbwoneill/ss100/S223r10b2.dat',
                 'rdkbwoneill/ss100/S223r4b1.dat',
                 'rdkbwoneill/ss100/S223r4b2.dat',
                 'rdkbwoneill/ss100/S223r5b1.dat',
                 'rdkbwoneill/ss100/S223r5b2.dat',
                 'rdkbwoneill/ss100/S223r6b1.dat',
                 'rdkbwoneill/ss100/S223r6b2.dat',
                 'rdkbwoneill/ss100/S223r7b1.dat',
                 'rdkbwoneill/ss100/S223r7b2.dat',
                 'rdkbwoneill/ss100/S223r8b1.dat',
                 'rdkbwoneill/ss100/S223r8b2.dat'}

# parallelize files into rddrdk_rdd = sc.parallelize(list_of_files)
rdk_rdd = sc.parallelize(list_of_files)

# rdd map: download file from s3, return each event
# rdd map: preprocess each channel (within channel, mean = 0, std = 1)
rdk_rdd = rdk_rdd.flatMap(dat_to_events)

# rdd map: maybe separate channels by type (sbd, bgo, bapd)
# build model (PowerIterationClustering)
# use PIC to train GradiantBoostedTrees
