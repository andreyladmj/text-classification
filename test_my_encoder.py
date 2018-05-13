import nltk
from nltk.tokenize import word_tokenize
import utils
import numpy as np

#nltk.download('punkt')

x_train = np.array([
    'Hello i am here',
    'Hello i am here 1',
    'Hello you no here',
    'Hello he am here',
    'hi i am there',
])

N = x_train.shape[0]
MAX_DOCUMENT_LENGTH = 0
words = list()
for sentence in x_train:
    tokenized = word_tokenize(sentence)
    print(tokenized)
    MAX_DOCUMENT_LENGTH = max(MAX_DOCUMENT_LENGTH, len(tokenized))
    words.extend(tokenized)

vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
print(vocab_to_int, int_to_vocab)

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


print(preprocessed_x_train)