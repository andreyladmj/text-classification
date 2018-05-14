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

from network_model import DBModel
from utils import create_lookup_tables


def train():
    db_model = DBModel()
    db_model.create_dataset()
    X_train, Y_train = db_model.get_train()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)


if __name__ == '__main__':
    train()