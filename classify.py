#from keras.datasets import mnist
#import matplotlib.pyplot as plt
#

#
#plt.subplot(221)
#plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
#
#plt.show()

import numpy
from keras.datasets import mnist
from keras.models improt Sequential
from keras.layers improt Dense
from keras.layers import Dropout
from keras.utils import np_utils

seed = 7
numpy.random.seed(seed)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

x_train = x_train / 255
x_test = x_test / 255


