import sys
import time
from collections import Counter
import tensorflow as tf
import numpy as np

from tensorflow.python.ops import rnn, rnn_cell
import pymysql
from tensorflow.python.ops.rnn import static_rnn
from tflearn import BasicLSTMCell
from tflearn.data_utils import VocabularyProcessor

from utils import create_lookup_tables


class DBModel():
    def __init__(self):
        local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
        self.db_cursor_read = local_db_connection.cursor()
        self.counter = Counter()
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self.max_sentence_length = 25

    def create_dataset(self):
        self.db_cursor_read.execute('SELECT * FROM prepared_orders where description_words = 25 ORDER BY rand()')
        for row in self.db_cursor_read: self.add(row)
        self.preprocess_data()

    def add(self, row):
        DB_STATUS_INDEX = 1
        DB_TITLE_INDEX = 5
        DB_ESTIMATED_INDEX = 3
        DB_DEADLINE_INDEX = 4
        DB_DESCRIPTION_INDEX = 6
        DB_IS_FIRST_ORDER_INDEX = 7

        state = row[DB_STATUS_INDEX]
        estimated = row[DB_ESTIMATED_INDEX]
        deadline = row[DB_DEADLINE_INDEX]
        first_order = row[DB_IS_FIRST_ORDER_INDEX]
        description_words = row[DB_DESCRIPTION_INDEX].split(' ')

        if state not in [3, 4, 5, 6, 9, 11]:
            return

        y = int(state in [3])
        words = []

        for word in description_words[:self.max_sentence_length]:
            word = self.prepare_word(word)
            self.add_word(word)
            words.append(word)

        x = [
            words,
            estimated,
            deadline,
            first_order
        ]

        self.X_train.append(words)
        self.Y_train.append([y])
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

    def prepare_word(self, word):
        word = word.strip("'")
        return word

    def add_word(self, word):
        self.counter[word] = self.counter.get(word, 0) + 1

    def preprocess_data(self):
        self.vocab_to_int, self.int_to_vocab = create_lookup_tables(self.counter)

        for row in range(len(self.X_train)):
            #self.X_train[row][0] = self.string_to_vocab(self.X_train[row][0], self.max_sentence_length)
            self.X_train[row] = self.string_to_vocab(self.X_train[row], self.max_sentence_length)

        test_data_size = 5000

        self.X_train = np.array(self.X_train, dtype=np.float)
        self.Y_train = np.array(self.Y_train, dtype=np.float)

        # self.X_test = self.X_train[:test_data_size]
        # self.Y_test = self.Y_train[:test_data_size]
        # self.X_train = self.X_train[test_data_size:]
        # self.Y_train = self.Y_train[test_data_size:]

        # self.X_test = np.array(self.X_train[:test_data_size], dtype=np.int32)
        # self.Y_test = np.array(self.Y_train[:test_data_size])
        #
        # self.X_train = np.array(self.X_train[test_data_size:], dtype=np.int32)
        # self.Y_train = np.array(self.Y_train[test_data_size:])

    def string_to_vocab(self, words, max_document_length):
        x = np.zeros(max_document_length)

        i = 0
        for word in words:
            x[i] = int(self.vocab_to_int[word])
            i += 1
            if i >= max_document_length:
                break

        return x

    def get_train(self):
        return self.X_train, self.Y_train

    def get_test(self):
        return self.X_test, self.Y_test