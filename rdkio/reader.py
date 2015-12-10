import struct
import numpy as np
import os

event_size = 8 * 2048 * 2 + 8 + 3 * 4 + 8
header_size = 5000


class RawData(object):
    """
    Data structure to store the RDK data

    :var start_flag: a set of 8 characters at the beginning of each event (should always be '********')

    :var index: number of the event in the file

    :var timestamp: when the event occurred (note: boards were not properly synchronized)

    :var n_channels: number of channels of data (should always be 8)

    :var hp_flag: I don't know why this is here, but it should always be 0xFF

    :var channel: numpy array (shape 8 x 2048) of signal traces
    """

    def __init__(self, data):
        """
        Initialize data structure

        :param data: An array of RDK data ( 8 characters, int, double, int, int, short[8][2048] )

        :return: self
        """
        if len(data) == 16396:
            self.start_flag = ''.join(data[:8])
            self.index = data[8]
            self.timestamp = data[9]
            self.n_channels = data[10]
            self.hp_flag = data[11]
            self.channel = np.array(data[12:]).reshape((8, 2048))


class FileReader(object):
    """
    Reads and formats data from RDK files

    :var filename: name of the RDK file

    :var file: file containing RDK data

    :var header: string containing header information from the RDK file

    :var footer: string containing footer information from the RDK file

    :var size: number of events in file
    """
    format = '=8cidii16384h'

    def __init__(self, filename=None):
        """
        Initialize reader

        :param filename: name of the RDK file

        :return: self
        """
        self.filename = filename
        self.file = None
        self.header = None
        self.footer = None
        self.size = None
        if filename is not None:
            self.open()

    def open(self, filename=None):
        """
        Open a new file

        :param filename: name of the RDK file

        :return: None
        """
        if self.file is not None:
            self.file.close()
        if filename is not None:
            self.filename = filename
        self.file = open(self.filename)
        self.header = self.file.read(header_size).rstrip('\x00')
        size = os.stat(self.filename).st_size
        self.size = (size - header_size) / event_size
        foot_size = size - event_size * self.size
        self.seek(self.size)
        self.footer = self.file.read(foot_size).rstrip('\x00')
        self.seek(0)

    def seek(self, index):
        """
        Set the file to the location of event number specified by index.

        :param index: integer number of the next event to read

        :return: None
        """
        self.file.seek(header_size + int(index) * event_size)

    def tell(self):
        """
        Return the number of the event at the files current position

        :return: current position (int)
        """
        return (self.file.tell() - header_size) / event_size

    def read(self):
        """
        Read the next event.

        :return: RawData object containing the data read from the file
        """
        data = self.file.read(event_size)
        data = struct.unpack(self.format, data)
        return RawData(data)
