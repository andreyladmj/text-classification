import sys
import time
from collections import Counter
import tensorflow as tf
import numpy as np

import pymysql
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
max_sentence_length = 500

def preprocess():
    db_cursor_read.execute('SELECT * FROM prepared_orders ORDER BY rand()')
    for row in db_cursor_read: preprocess_row(row)
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
        add_word(word)
        x.append(word)

    X_train.append([' '.join(x)])
    Y_train.append(int(state in [3]))

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
    X_test = np.array(X_train[:test_data_size])
    Y_test = np.array(Y_train[:test_data_size])

    X_train = np.array(X_train[test_data_size:])
    Y_train = np.array(Y_train[test_data_size:])


def string_to_vocab(string, max_document_length):
    x = np.zeros(max_document_length)

    i = 0
    for word in string.split(' '):
        x[i] = int(vocab_to_int[word])
        i += 1
        if i >= max_document_length:
            break

    return x

# def preprocess_data():
#     vocab_processor = VocabularyProcessor(max_document_length=max_sentence_length, vocabulary=vocab_to_int)
#     for row in range(len(X_train)):
#         print(row)
#         print(X_train[row])
#
#         for i in X_train[row][0].split(' '):
#             print(vocab_to_int[i], end='+ ')
#
#         X_train[row] = list(vocab_processor.transform(X_train[row]))
#         print(X_train[row])
#     raise Exception

def build_output(lstm_output, in_size, out_size):
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

epochs = 20

def create_neural_network():
    global vocab_to_int, int_to_vocab
    vocab_to_int, int_to_vocab = create_lookup_tables(counter)
    preprocess_data()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_test', X_test.shape)
    print('Y_test', Y_test.shape)
    print('size of vocabulary', len(vocab_to_int)) #124188

    sequence_length = max_sentence_length
    embedding_length = len(vocab_to_int)
    num_classes = 2

    print('sequence_length', sequence_length)
    print('embedding_length', embedding_length)

    input_data = tf.placeholder(tf.float32, [None, sequence_length, embedding_length])


    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    hidden_vector_size = 100

    rnn_cell = tf.contrib.rnn.LSTMCell(hidden_vector_size)

    initial_zero_h = tf.matmul(tf.reduce_mean(tf.zeros_like(input_data), 2),
                               tf.zeros([sequence_length, hidden_vector_size]))

    initial_state = tf.contrib.rnn.LSTMStateTuple(initial_zero_h, initial_zero_h)

    outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                       initial_state=initial_state,
                                       dtype=tf.float32)


    prediction, logits = build_output(outputs, hidden_vector_size, num_classes)

    loss = build_loss(logits, targets, hidden_vector_size, num_classes)
    optimizer = build_optimizer(loss, learning_rate, grad_clip)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    #saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(tf.global_variables_initializer())
        loss = 0
        for x, y in get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {input_data: x,
                    model.targets: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state}
            batch_loss, new_state, _ = sess.run([model.loss,
                                                 model.final_state,
                                                 model.optimizer],
                                                feed_dict=feed)

            end = time.time()
            print('Epoch: {}/{}... '.format(e+1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end-start)))




    # batch_size = 14
    # fake_input = np.random.uniform(size=[batch_size,96,64])
    # outputs.eval(feed_dict={input_data:fake_input})[0,:10,:10]
    #
    # batch_size = 140
    # fake_input = np.random.uniform(size=[batch_size,96,64])
    # outputs.eval(feed_dict={input_data:fake_input})[0,:10,:10]


if __name__ == '__main__':
    preprocess()