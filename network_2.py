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
    y = int(state in [3])
    Y_train.append([y, 1 - y])

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
    X_test = np.array(X_train[:test_data_size], dtype=np.int32)
    Y_test = np.array(Y_train[:test_data_size])

    X_train = np.array(X_train[test_data_size:], dtype=np.int32)
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

def build_inputs(batch_size, num_steps):
    inputs = tf.placeholder(tf.int32, [None, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [None, 2], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, keep_prob

def build_output(lstm_output, in_size, out_size):
    print('---------------------')
    print('in_size', in_size)
    print('out_size', out_size)
    print('lstm_output', lstm_output.shape)
    seq_output = tf.concat(lstm_output, axis=1)
    print('seq_output', seq_output.shape)
    x = tf.reshape(seq_output, [-1, in_size])
    print('x', x.shape)

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x, softmax_w) + softmax_b
    print('logits size', logits.shape)
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits


def build_loss(logits, targets, lstm_size, num_classes):
    # y_one_hot = tf.one_hot(targets, num_classes)
    #
    print('targets', targets, targets.shape)
    print('logits', logits, logits.get_shape())
    # print('num_classes', num_classes)
    # print('y_one_hot', y_one_hot)
    # y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = tf.reduce_mean(loss)
    return loss



def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    def build_cell(num_units, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        return drop

    ### Build the LSTM Cell
    # Use a basic LSTM cell
    print('lstm_size', lstm_size)
    print('batch_size', batch_size)
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    # Add dropout to the cell outputs
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    #cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, initial_state



def build_optimizer(loss, learning_rate, grad_clip):
    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def get_batches(X_train, batch_size, num_steps):
    pass

def create_neural_network():
    global vocab_to_int, int_to_vocab, counter
    vocab_to_int, int_to_vocab = create_lookup_tables(counter)
    preprocess_data()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_test', X_test.shape)
    print('Y_test', Y_test.shape)
    print('size of vocabulary', len(vocab_to_int))

    time_steps=128
    num_units=128 #hidden LSTM units

    n_input=500 #rows of 28 pixels

    learning_rate=0.001 #learning rate for adam

    n_classes=2 #mnist is meant to be classified in 10 classes(0-9).

    batch_size=128 #size of batch

    tf.reset_default_graph()

    out_weights = tf.Variable(tf.random_normal([n_input, n_classes]))
    out_bias = tf.Variable(tf.random_normal([n_classes]))

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    #input = tf.unstack(x, n_input, 0)

    lstm_layer = BasicLSTMCell(num_units, forget_bias=1)
    outputs, _ = rnn.rnn(lstm_layer, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], out_weights)+out_bias

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #model evaluation
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        iter=1
        while iter<800:
            for batch_x, batch_y in batch_features_labels(X_train, Y_train, batch_size):
            #batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
                print('batch_x', batch_x.shape)
                print('batch_y', batch_y.shape)

                #batch_x = batch_x.reshape((batch_size,time_steps,n_input))

                sess.run(opt, feed_dict={x: batch_x, y: batch_y})

                if iter %10==0:
                    acc=sess.run(accuracy,feed_dict={x:batch_x, y:batch_y})
                    los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
                    print("For iter ", iter)
                    print("Accuracy ", acc)
                    print("Loss ", los)
                    print("__________________")

                iter=iter+1

if __name__ == '__main__':
    preprocess()