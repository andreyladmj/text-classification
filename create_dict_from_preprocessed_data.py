import sys
from collections import Counter

import pymysql

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()
db_cursor_write = local_db_connection.cursor()

#db_cursor_write.execute('TRUNCATE TABLE dict;')
counter = Counter()


def preprocess():
    db_cursor_read.execute('SELECT * FROM prepared_orders')
    c = 0
    for row in db_cursor_read:
        c += 1
        preprocess_row(row)
        print('\rProcessed: {} rows'.format(c), end='')

    print('\nTotal:', c)
    print(counter.most_common(50))

    local_db_connection.commit()


def preprocess_row(row):
    DB_TITLE_INDEX = 5
    DB_DESCRIPTION_INDEX = 6

    title_words = row[DB_TITLE_INDEX].split(' ')
    description_words = row[DB_DESCRIPTION_INDEX].split(' ')

    for word in title_words:
        add_word(word)

    for word in description_words:
        add_word(word)


def add_word(word):
    global counter
    word = word.strip("'")
    counter[word] = counter.get(word, 0) + 1


def add_word_to_db(word):
    word = word.strip("'")

    try:
        query = 'SELECT * FROM dict WHERE word = "{}";'.format(word)
        res = db_cursor_write.execute(query)

        if res:
            query = 'UPDATE dict SET count = count + 1 WHERE word = "{}";'.format(word)
        else:
            query = 'INSERT INTO dict (word) values ("{}");'.format(word)

        db_cursor_write.execute(query)
    except Exception as e:
        print(query)
        raise e



def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """

    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {(ii+1): word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: (ii+1) for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

# '1', 'Bidding', NULL, '1'
# '2', 'In Progress', NULL, '1'
# '3', 'Finished', NULL, '1'
# '4', 'Canceled by Customer', NULL, '1'
# '5', 'Canceled by Writer', NULL, '1'
# '6', 'Canceled by System', NULL, '1'
# '9', 'Not Active', NULL, '2'
# '10', 'Active', NULL, '2'
# '11', 'Expired', NULL, '2'
# '12', 'Pending Payment', NULL, '1'
# '13', 'Under Investigation', NULL, '1'
# '14', 'Pending Writer', NULL, '1'


if __name__ == '__main__':
    preprocess()