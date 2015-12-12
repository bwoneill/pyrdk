import psycopg2 as psql
from reader import *

create_tbl = '''CREATE TABLE data (
    type char,
    series int,
    run int,
    board int,
    index int,
    timestamp float4,
    channels int2[8][2048]
)
'''

add_event = '''INSERT INTO data VALUES (%s)'''


def array_to_sql(array):
    temp = ['{' + ','.join(row.astype(str)) + '}' for row in array]
    result = '{' + ','.join(temp) + '}'
    return result


class SqlManager(object):
    def __init__(self):
        self.conn = psql.connect(database='rdk', user='bwoneill')
        self.conn.set_session(autocommit=True)
        self.curs = self.conn.cursor()
        self.curs.execute("select * from information_schema.tables where table_name='data'")
        if not bool(self.curs.rowcount):
            self.curs.execute(create_tbl)
            self.conn.commit()

    def add_event(self, data):
        self.curs.execute(add_event % str(data))
        return self.curs.statusmessage

    def commit(self):
        self.conn.commit()
