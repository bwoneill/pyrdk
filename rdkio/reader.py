import struct
import numpy as np
import os
import re
import boto3

event_size = 8 * 2048 * 2 + 8 + 3 * 4 + 8
header_size = 5000

s3 = boto3.resource('s3')

class RawData(object):
    """
    Data structure to store the RDK data

    :var start_flag: a set of 8 characters at the beginning of each event (should always be '********')

    :var type: Type of the data file event if from

    :var series: Number of the series this data was taken in

    :var run: Number of the run this data was taken in

    :var board: Number of the board this data was recorded on

    :var index: number of the event in the file

    :var timestamp: when the event occurred (note: boards were not properly synchronized)

    :var n_channels: number of channels of data (should always be 8)

    :var hp_flag: I don't know why this is here, but it should always be 0xFF

    :var signal: numpy array (shape 8 x 2048) of signal traces
    """

    def __init__(self, t, series, run, board, data):
        """
        Initialize data structure

        :param t: Type of the data file event if from

        :param series: Number of the series this data was taken in

        :param run: Number of the run this data was taken in

        :param board: Number of the board this data was recorded on

        :param data: An array of RDK data ( 8 characters, int, double, int, int, short[8][2048] )

        :return: self
        """
        self.type = t
        self.series = series
        self.run = run
        self.board = board
        if len(data) == 16396:
            self.start_flag = ''.join(data[:8])
            self.index = data[8]
            self.timestamp = data[9]
            self.n_channels = data[10]
            self.hp_flag = data[11]
            self.signal = np.array(data[12:]).reshape((8, 2048)).astype(float)

    def __repr__(self):
        result = '\'%s\', %i, %i, %i, %i, %f, %s' % (
            self.type, self.series, self.run, self.board, self.index, self.timestamp,
            '\'{' + ','.join(['{' + ','.join(row.astype(str)) + '}' for row in self.signal]) + '}\'')
        return result


class FileReader(object):
    """
    Reads and formats data from RDK files

    :var filename: name of the RDK file

    :var file: file containing RDK data

    :var type: from filename

    :var series: from filename

    :var run: from filename

    :var board: from filename

    :var header: string containing header information from the RDK file

    :var footer: string containing footer information from the RDK file

    :var size: number of events in file
    """
    data_format = '=8cidii16384h'
    filename_format = r'([SCD])(\d+)r(\d+)b(\d+)'
    s3_format = r'([^/]+)/(.*)'

    def __init__(self, filename=None, source='source'):
        """
        Initialize reader

        :param filename: name of the RDK file

        :param source: 'local' or 's3'

        :return: self
        """
        self.filename = filename
        self.file = None
        self.header = None
        self.footer = None
        self.size = None
        self.type = None
        self.series = None
        self.run = None
        self.board = None
        self.source = source
        self.bucket = None
        self.key = None
        if filename is not None:
            self.open()

    def open(self, filename=None, source=None):
        """
        Open a new file

        :param filename: name of the RDK file

        :param source: 'local' or 's3'

        :return: None
        """
        if self.file is not None:
            self.file.close()
        if filename is not None:
            self.filename = filename
        if source is not None:
            self.source = source
        temp = re.search(self.filename_format, self.filename).groups()
        self.type = temp[0]
        self.series = int(temp[1])
        self.run = int(temp[2])
        self.board = int(temp[3])
        if self.source == 'source':
            self.file = open(self.filename)
            size = os.stat(self.filename).st_size
        elif self.source == 's3':
            self.bucket, self.key = re.search(self.s3_format, self.filename).groups()
            obj = s3.Object(self.bucket, self.key)
            temp = obj.get()
            size = temp['ContentLength']
            self.file = temp['Body']
        self.size = (size - header_size) / event_size
        foot_size = size - event_size * self.size
        self.seek(self.size)
        self.footer = self.file.read(foot_size).rstrip('\x00')
        self.header = self.file.read(header_size).rstrip('\x00')
        self.seek(0)

    def seek(self, index):
        """
        Set the file to the location of event number specified by index.

        :param index: integer number of the next event to read

        :return: None
        """
        loc = header_size + int(index) * event_size
        if self.source:
            self.file.seek(loc)
        else:
            obj = s3.Object(self.bucket, self.key)
            self.file = obj.get(Range='bytes=%i-' % loc)['Body']

    def tell(self):
        """
        Return the number of the event at the files current position

        :return: current position (int)
        """
        if self.source:
            return (self.file.tell() - header_size) / event_size
        else:
            return -1

    def read(self):
        """
        Read the next event.

        :return: RawData object containing the data read from the file
        """
        data = self.file.read(event_size)
        if len(data) == event_size:
            data = struct.unpack(self.data_format, data)
            return RawData(self.type, self.series, self.run, self.board, data)
        else:
            return None
