import sys
from collections import Counter
import tensorflow as tf

import pymysql

from utils import create_lookup_tables

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()

counter = Counter()


def preprocess():
    db_cursor_read.execute('SELECT * FROM prepared_orders')
    for row in db_cursor_read: preprocess_row(row)
    db_cursor_read.execute("SELECT max(name_words), max(description_words) FROM prepared_orders")
    max_name_words, max_description_words = db_cursor_read.fetchone()
    create_neural_network(max_description_words)


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


def create_neural_network(max_length):
    #https://github.com/KnHuq/Dynamic-Tensorflow-Tutorial
    vocab_to_int, int_to_vocab = create_lookup_tables(counter)

    sequence_length = 96
    embedding_length = 64

    input_data = tf.placeholder(tf.float32, [None, sequence_length, embedding_length])

    hidden_vector_size = 100

    rnn_cell = tf.contrib.rnn.LSTMCell(hidden_vector_size)

    initial_zero_h = tf.matmul(tf.reduce_mean(tf.zeros_like(input_data), 2),
                               tf.zeros([sequence_length, hidden_vector_size]))

    initial_state = tf.contrib.rnn.LSTMStateTuple(initial_zero_h, initial_zero_h)

    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                       initial_state=initial_state,
                                       dtype=tf.float32)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    batch_size = 14
    fake_input = np.random.uniform(size=[batch_size,96,64])
    outputs.eval(feed_dict={input_data:fake_input})[0,:10,:10]

    batch_size = 140
    fake_input = np.random.uniform(size=[batch_size,96,64])
    outputs.eval(feed_dict={input_data:fake_input})[0,:10,:10]


if __name__ == '__main__':
    preprocess()