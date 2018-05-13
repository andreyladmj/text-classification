from tensorflow.contrib import learn
import numpy as np
import pandas as pd

x_train = pd.DataFrame([
    ['Hello i am here'],
    ['Hello i am here 1'],
    ['Hello you am here'],
    ['Hello he am here'],
    ['hi i am there'],
])[1]
x_test = np.ones((5))

print(x_train, x_train.shape)
print(x_test, x_test.shape)

MAX_DOCUMENT_LENGTH = 6

vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

print(list(vocab_processor.fit_transform(x_train)))
print(list(vocab_processor.transform(x_test)))
raise Exception

x_train = np.array(list(vocab_processor.fit_transform(x_train)))
x_test = np.array(list(vocab_processor.transform(x_test)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)
print(x_train, x_train.shape)
print(x_test, x_test.shape)
