import pandas as pd
from rdkio.reader import *
from rdkio.sql import *
import numpy as np
import psycopg2 as psql

if __name__ == '__main__':
    reader = FileReader('/mnt/shared/ss223/S223r4b1.dat')
    manager = SqlManager()
    while reader.tell() < reader.size:
        data = reader.read()
        manager.add_event(data)
        if reader.tell() % 1000 == 0:
            print reader.tell()
    manager.commit()
