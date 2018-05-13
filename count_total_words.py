import sys

import pymysql

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()
db_cursor_write = local_db_connection.cursor()

db_cursor_write.execute('TRUNCATE TABLE dict;')

MAX_TITLE_WORDS = 0
MAX_DESCRIPTION_WORDS = 0


def preprocess():
    db_cursor_read.execute('SELECT * FROM prepared_orders')
    c = 0
    for row in db_cursor_read:
        c += 1
        preprocess_row(row)
        print('\rProcessed: {} rows'.format(c), end='')

    print('\nTotal:', c)
    print('MAX_TITLE_WORDS:', MAX_TITLE_WORDS)
    print('MAX_DESCRIPTION_WORDS:', MAX_DESCRIPTION_WORDS)
    local_db_connection.commit()


def preprocess_row(row):
    global MAX_TITLE_WORDS, MAX_DESCRIPTION_WORDS

    DB_TITLE_INDEX = 5
    DB_DESCRIPTION_INDEX = 6

    title_words = row[DB_TITLE_INDEX].split(' ')
    description_words = row[DB_DESCRIPTION_INDEX].split(' ')
    MAX_TITLE_WORDS = max(MAX_TITLE_WORDS, len(title_words))
    MAX_DESCRIPTION_WORDS = max(MAX_DESCRIPTION_WORDS, len(description_words))


if __name__ == '__main__':
    preprocess()