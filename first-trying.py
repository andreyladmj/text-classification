from sklearn.preprocessing import OneHotEncoder
from tensorflow.contrib import learn
import pandas as pd
import numpy as np
import itertools
from collections import Counter
import utils
from nltk.tokenize import word_tokenize

dbpedia = learn.datasets.load_dataset('dbpedia')

print(type(dbpedia), dbpedia.__slots__)

x_train = pd.DataFrame(dbpedia.train.data)[1]
y_train = pd.Series(dbpedia.train.target)
x_test = pd.DataFrame(dbpedia.test.data)[1]
y_test = pd.Series(dbpedia.test.target)
#x_dev = pd.DataFrame(dbpedia.validation.data)[1]
#y_dev = pd.Series(dbpedia.validation.target)

print('x_train shape', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

N = x_train.shape[0]
MAX_DOCUMENT_LENGTH = 0
words = list()
for sentence in x_train:
    tokenized = word_tokenize(sentence)
    MAX_DOCUMENT_LENGTH = max(MAX_DOCUMENT_LENGTH, len(tokenized))
    words.extend(tokenized)

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)

preprocessed_x_train = np.zeros((N, MAX_DOCUMENT_LENGTH))

row = 0
for sentence in x_train:
    tokenized = word_tokenize(sentence)
    #preprocessed_x_train.append([vocab_to_int[word] for word in tokenized])
    col = 0
    for word in tokenized:
        preprocessed_x_train[row, col] = vocab_to_int[word]
        col += 1
    row += 1


#print("Total words: {}".format(len(words)))
#print("Unique words: {}".format(len(set(words))))

#preprocessed_x_train = np.array(preprocessed_x_train)

#print(x_train)
#print(preprocessed_x_train, preprocessed_x_train.shape)


vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)