from __future__ import division

import collections
import numpy as np
import tensorflow
import dong.framework

DataPair = collections.namedtuple('DataPair', ['x', 'y'])
DataParams = collections.namedtuple('Params', ['shape', 'num_labels'])

class Cifar10(dong.framework.Data):

    
    def __init__(self, config=None):

        data = tensorflow.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = data.load_data()

        self._num_labels = len(np.unique(y_train))
        # transform y to 1-hot.
        self._y_train = tensorflow.keras.utils.to_categorical(y_train, self._num_labels).astype('float32')
        self._y_test = tensorflow.keras.utils.to_categorical(y_test, self._num_labels).astype('float32')
        # normalize to values of the pixels.
        self._x_train, self._x_test = x_train / 255.0, x_test / 255.0
        self._x_train = self._x_train.astype('float32')
        self._x_test = self._x_test.astype('float32')
        
    def get_train_data(self):
        return DataPair(self._x_train, self._y_train)

    def get_eval_data(self):
        return DataPair(self._x_test, self._y_test)

    def get_data_params(self):
        self._shape = self._x_train.shape[1:]
        return DataParams(self._shape, self._num_labels)


    
