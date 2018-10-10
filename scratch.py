import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def softmax(x):
	expX = np.exp(x)
	return expX / expX.sum(axis=1, keepdims=True)

def relu(x):
	if x<0:
		return 0
	else:
		return x

# M is the number of hidden layers
#TODO define M, D, and K

#x_train is 60000 samples, 28x28 pixel images
#y_train is the 60000 targets, with values 0-9
#x_test is 10000 samples
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]
print(num_pixels)
#Flatten into Nx784 arrays
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
print(x_train)
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#Normalize pixel intensity values
x_train = x_train / 255
x_test = x_test / 255


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

W = np.random.randn(D,M)
V = np.random.randn(M,K)
Z = sigmoid(X.dot(W))
p_y_given_x = softmax(Z.dot(V))


