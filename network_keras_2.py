import sys
import time
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from tensorflow.python.ops import rnn, rnn_cell
import pymysql
from tensorflow.python.ops.rnn import static_rnn
from tflearn import BasicLSTMCell
from tflearn.data_utils import VocabularyProcessor

from utils import create_lookup_tables

local_db_connection = pymysql.connect(host='localhost', database='edusson', user='root', password='')
db_cursor_read = local_db_connection.cursor()

counter = Counter()
X_train = []
Y_train = []
X_test = []
Y_test = []
vocab_to_int = {}
int_to_vocab = {}
max_sentence_length = 100
bad_words = []

def preprocess():
    db_cursor_read.execute('SELECT * FROM prepared_orders ORDER BY rand()')
    for row in db_cursor_read: preprocess_row(row)
    #print(counter.most_common(100))
    #raise EnvironmentError
    #db_cursor_read.execute("SELECT max(name_words), max(description_words) FROM prepared_orders")
    #max_name_words, max_description_words = db_cursor_read.fetchone()
    create_neural_network()


def preprocess_row(row):
    global X_train, Y_train

    DB_STATUS_INDEX = 1
    DB_TITLE_INDEX = 5
    DB_DESCRIPTION_INDEX = 6

    state = row[DB_STATUS_INDEX]
    title_words = row[DB_TITLE_INDEX].split(' ')
    description_words = row[DB_DESCRIPTION_INDEX].split(' ')

    if state not in [3, 4, 5, 6, 9, 11]:
        return

    # for word in title_words:
    #     add_word(word)

    x = []

    for word in description_words[:max_sentence_length]:
        word = prepare_word(word)
        if word:
            add_word(word)
            x.append(word)

    X_train.append([' '.join(x)])
    y = int(state in [3])
    Y_train.append([y])

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

def prepare_word(word):
    word = word.strip("'")
    if len(word) < 4: return None

    if word in bad_words:
        return None

    return word

def add_word(word):
    global counter
    counter[word] = counter.get(word, 0) + 1

def preprocess_data():
    global X_train, Y_train, X_test, Y_test

    #vocab_processor = VocabularyProcessor(max_document_length=max_sentence_length, vocabulary=vocab_to_int)
    for row in range(len(X_train)):
        encoded = string_to_vocab(X_train[row][0], max_sentence_length)
        X_train[row] = encoded
        #X_train[row] = list(vocab_processor.transform(X_train[row]))
        #print(X_train[row])

    test_data_size = 5000
    X_test = np.array(X_train[:test_data_size], dtype=np.int32)
    Y_test = np.array(Y_train[:test_data_size])

    X_train = np.array(X_train[test_data_size:], dtype=np.int32)
    Y_train = np.array(Y_train[test_data_size:])


def string_to_vocab(string, max_document_length):
    x = np.zeros(max_document_length)

    i = 0
    for word in string.split(' '):
        #x[i] = int(vocab_to_int[word])
        x[i] = vocab_to_int.get(word, 0)
        i += 1
        if i >= max_document_length:
            break

    return x

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def create_neural_network():
    global vocab_to_int, int_to_vocab, counter
    vocab_to_int, int_to_vocab = create_lookup_tables(counter)
    preprocess_data()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_test', X_test.shape)
    print('Y_test', Y_test.shape)
    print('size of vocabulary', len(vocab_to_int))
    model = RNN()
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=128,epochs=10,
              validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

max_len = 100
max_words = 77675#110565

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

if __name__ == '__main__':
    preprocess()