import cProfile
import line_profiler
from timeit import Timer

import pymysql

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()


@profile
def benchmark():
    db_cursor_read.execute('SELECT * FROM es_orders')
    c = 0
    for row in db_cursor_read:
        c+=1
    return c


# pr = cProfile.Profile()
# pr.enable()
benchmark()
# pr.disable()
# pr.print_stats()
#kernprof -l script_to_profile.py
#python -m line_profiler test.py.lprof