import tensorflow as tf

import numpy as np
from tensorflow.python.ops import rnn
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell, LSTMCell

from utils import batch_features_labels


class Network_model:
    def __init__(self, num_input, num_classes, num_hidden):
        self.num_input = num_input
        self.num_classes = num_classes
        self.num_hidden = num_hidden

    def create_placeholders(self):
        self.input = tf.placeholder("float", [None, self.num_input], name = 'input')
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name='labels')
        self.keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    def lstm_setup(self, input):
        with tf.name_scope("LSTM_setup") as scope:
            def single_cell():
                return tf.contrib.rnn.DropoutWrapper(LSTMCell(self.num_hidden), output_keep_prob=self.keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            initial_state = cell.zero_state(batch_size, tf.float32)

        input_list = tf.unstack(tf.expand_dims(input, axis=2), axis=1)
        outputs, _ = static_rnn(cell, input_list, dtype=tf.float32)
        #outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)

        output = outputs[-1]

    def softmax(self):
        with tf.name_scope("Softmax") as scope:
            with tf.variable_scope("Softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [num_hidden, num_classes])
                softmax_b = tf.get_variable("softmax_b", [num_classes])

            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
            #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
            cost = tf.reduce_sum(loss) / batch_size
            # For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size]
            # and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
            # For softmax_cross_entropy_with_logits, labels must have the shape
            # [batch_size, num_classes] and dtype float32 or float64.

    def evaluate(self):
        with tf.name_scope("Evaluating_accuracy") as scope:
            print('logits', logits)
            print('labels', labels)
            print('tf.argmax(logits, 1)', tf.argmax(logits, 1))
            #correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            predicted = tf.round(tf.nn.sigmoid(logits))
            correct_prediction = tf.equal(predicted, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            #correct_prediction = tf.equal(logits, labels)
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            h1 = tf.summary.scalar('accuracy', accuracy)
            h2 = tf.summary.scalar('cost', cost)

    def optimizer(self):
        with tf.name_scope("Optimizer") as scope:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)   #We clip the gradients to prevent explosion
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = zip(grads, tvars)
            train_op = optimizer.apply_gradients(gradients)

    def run(self):
        tf.summary.merge_all()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        try:
            for i in range(100):
                print('Iteration', i)
                for batch_x, batch_y in batch_features_labels(X_train, Y_train, batch_size):
                    logits_, labels_, correct_prediction_, accuracy_, predicted_,  cost_train, acc_train, _ = sess.run([
                        logits, labels, correct_prediction, accuracy, predicted,
                        cost,
                        accuracy,
                        train_op
                    ], feed_dict = {input: batch_x, labels: batch_y, keep_prob:0.5})
                    #cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
                    #acc_train_ma = acc_train_ma*0.99 + acc_train*0.01

                    #if i%4 == 1:
                    # print('---------')
                    print('cost_train', cost_train, 'acc_train', acc_train)

                    if i > 2:
                        test(sess, predicted, keep_prob, db_model, num_input, input)
                    # print('logits_')
                    # print(logits_)
                    # print('predicted')
                    # print(predicted_)
                    # print('labels_')
                    # print(labels_)
                    # print('correct_prediction_')
                    # print(correct_prediction_)
                    # print('accuracy_')
                    # print(accuracy_)
                    # print('---------')
                    #Evaluate validation performance
                    #X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
                    #cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
                    #print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
        except KeyboardInterrupt:
            print('Catched KeyboardInterrupt')

    def test(self):
        def test(sess, predicted, keep_prob, db_model, num_input, input):
            test = np.array([
                'provide adequate citation for each of the references in the bibliography . none of the sources can be websites or web pages or documents .',
                'mla format biography research on maya angelou .. 8-10 pages work cited page with 5 in text citation and 5 work citiation at the end',
                'i have $ 4,60 and would like a decent paper written , a b level english paper for senior rhetoric class , senior class .',
                'does the student have a well-thought out introduction that makes sense and effectively helps the reader understand the topic in an organized way ? 10',
                '1. description of the importance of the topic 2.review of the research in relation to the topic 3.compare and contrast the research of the topic',
                'instruction : this is not an essay , but it is a form of question and answer . the three questions are listed below .',
                'sources most be : 1 primary article , 1 journal article , 1 encyclopedia article and 1 book . also it most have an outline',
                'this is a research essay that needs to be re-written in order for it to flow better . mla format . 1250 words minimum .',
            ])

            x = []

            for i in range(len(test)):
                x.append(db_model.string_to_vocab(test[i].split(' '), num_input))

            x = np.array(x)

            # 3 1
            # 4 0
            # 6 0
            # 6 0
            # 3 1
            # 3 1
            # 6 0
            # 4 0


            # print('test: ', test, test.shape)
            # print('x: ', x, x.shape)
            predicted_ = sess.run(predicted, feed_dict = {input: x, keep_prob: 1.0})
            print('predicted_', predicted_)

def train():
    db_model = DBModel()
    db_model.create_dataset()
    X_train, Y_train = db_model.get_train()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)

    learning_rate = 0.001
    training_steps = 10000
    batch_size = 128
    display_step = 10
    max_grad_norm = 5

    # Network Parameters
    num_input = 25 # MNIST data input (img shape: 28*28)
    timesteps = 28 # timesteps
    num_hidden = 128 # hidden layer num of features
    num_classes = 1 # MNIST total classes (0-9 digits)
    num_layers = 2

    # tf Graph input
    # X = tf.placeholder("float", [None, num_input])
    # Y = tf.placeholder("float", [None, num_classes])
    # keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    input = tf.placeholder("float", [None, num_input], name = 'input')
    labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
    keep_prob = tf.placeholder("float", name = 'Drop_out_keep_prob')

    with tf.name_scope("LSTM_setup") as scope:
        def single_cell():
            return tf.contrib.rnn.DropoutWrapper(LSTMCell(num_hidden),output_keep_prob=keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)

    input_list = tf.unstack(tf.expand_dims(input, axis=2), axis=1)
    outputs, _ = static_rnn(cell, input_list, dtype=tf.float32)

    print('outputs', outputs)

    output = outputs[-1]


    with tf.name_scope("Softmax") as scope:
        with tf.variable_scope("Softmax_params"):
            softmax_w = tf.get_variable("softmax_w", [num_hidden, num_classes])
            softmax_b = tf.get_variable("softmax_b", [num_classes])

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='softmax')
        cost = tf.reduce_sum(loss) / batch_size
        # For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size]
        # and the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
        # For softmax_cross_entropy_with_logits, labels must have the shape
        # [batch_size, num_classes] and dtype float32 or float64.

    with tf.name_scope("Evaluating_accuracy") as scope:
        print('logits', logits)
        print('labels', labels)
        print('tf.argmax(logits, 1)', tf.argmax(logits, 1))
        #correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        predicted = tf.round(tf.nn.sigmoid(logits))
        correct_prediction = tf.equal(predicted, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        #correct_prediction = tf.equal(logits, labels)
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        h1 = tf.summary.scalar('accuracy', accuracy)
        h2 = tf.summary.scalar('cost', cost)

    """Optimizer"""
    with tf.name_scope("Optimizer") as scope:
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)   #We clip the gradients to prevent explosion
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = zip(grads, tvars)
        train_op = optimizer.apply_gradients(gradients)

    tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    try:
        for i in range(100):
            print('Iteration', i)
            for batch_x, batch_y in batch_features_labels(X_train, Y_train, batch_size):
                logits_, labels_, correct_prediction_, accuracy_, predicted_,  cost_train, acc_train, _ = sess.run([
                    logits, labels, correct_prediction, accuracy, predicted,
                    cost,
                    accuracy,
                    train_op
                ], feed_dict = {input: batch_x, labels: batch_y, keep_prob:0.5})
                #cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
                #acc_train_ma = acc_train_ma*0.99 + acc_train*0.01

                #if i%4 == 1:
                # print('---------')
                print('cost_train', cost_train, 'acc_train', acc_train)

                if i > 2:
                    test(sess, predicted, keep_prob, db_model, num_input, input)
                # print('logits_')
                # print(logits_)
                # print('predicted')
                # print(predicted_)
                # print('labels_')
                # print(labels_)
                # print('correct_prediction_')
                # print(correct_prediction_)
                # print('accuracy_')
                # print(accuracy_)
                # print('---------')
                    #Evaluate validation performance
                    #X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
                    #cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
                    #print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
    except KeyboardInterrupt:
        print('Catched KeyboardInterrupt')


def test(sess, predicted, keep_prob, db_model, num_input, input):
    test = np.array([
        'provide adequate citation for each of the references in the bibliography . none of the sources can be websites or web pages or documents .',
        'mla format biography research on maya angelou .. 8-10 pages work cited page with 5 in text citation and 5 work citiation at the end',
        'i have $ 4,60 and would like a decent paper written , a b level english paper for senior rhetoric class , senior class .',
        'does the student have a well-thought out introduction that makes sense and effectively helps the reader understand the topic in an organized way ? 10',
        '1. description of the importance of the topic 2.review of the research in relation to the topic 3.compare and contrast the research of the topic',
        'instruction : this is not an essay , but it is a form of question and answer . the three questions are listed below .',
        'sources most be : 1 primary article , 1 journal article , 1 encyclopedia article and 1 book . also it most have an outline',
        'this is a research essay that needs to be re-written in order for it to flow better . mla format . 1250 words minimum .',
    ])

    x = []

    for i in range(len(test)):
        x.append(db_model.string_to_vocab(test[i].split(' '), num_input))

    x = np.array(x)

# 3 1
# 4 0
# 6 0
# 6 0
# 3 1
# 3 1
# 6 0
# 4 0


    # print('test: ', test, test.shape)
    # print('x: ', x, x.shape)
    predicted_ = sess.run(predicted, feed_dict = {input: x, keep_prob: 1.0})
    print('predicted_', predicted_)


if __name__ == '__main__':
    train()