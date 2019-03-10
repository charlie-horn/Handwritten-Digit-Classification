import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import math
import cv2
from scipy.misc import imread,imsave


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

def cost(T,Y):
  return -(T*np.log(Y)).sum()

#weight = weight - learning_rate*gradient

def forward(X, W1, W2):
  Z = sigmoid(X.dot(W1))
  Y = softmax(Z.dot(W2))
  return Y,Z

def grad_W2(Z,T,Y):
  return Z.T.dot(Y - T)

def grad_W1(X,Z,T,Y,W2):
  return X.T.dot(((Y - T).dot(W2.T)*(Z*(1 - Z))))

def draw(event, x, y):
  global img, drawing
  if event == cv2.EVENT_LBUTTONDOWN:
    img[x,y] = 0


#x_train is 60000 samples, 28x28 pixel images
#y_train is the 60000 targets, with values 0-9
#x_test is 10000 samples
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]

#Flatten into Nx784 arrays
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

#Normalize pixel intensity values
x_train = x_train / 255
x_test = x_test / 255

#Transform y into index vectors
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]



# M is the size of hidden layer
# K is the number of classes
# D is the size of a sample
D = num_pixels
K = num_classes
M = D


# Instantiate Weights
W1 = np.random.randn(D,M)
W2 = np.random.randn(M,K)

Z = sigmoid(x_train.dot(W1))
Y = softmax(Z.dot(W2))

#Training

epochs = 10
learning_rate = 0.0000000001
C = 0

#while(not math.isnan(C)):
for i in range(epochs):
    Y, Z = forward(x_train, W1, W2)
    W2 -= learning_rate*grad_W2(Z, y_train, Y)
    W1 -= learning_rate*grad_W1(x_train, Z, y_train, Y, W2)
    C = cost(y_train, Y)
    print(C)
#    learning_rate = learning_rate*10
#    print("New rate : " + str(learning_rate))

#Testing

Y, Z = forward(x_test, W1, W2)
successes = 0
failures = 0
hypothesis = 0
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        if Y[i][j] > Y[i][hypothesis]:
            hypothesis = j
    for j in range(y_test.shape[0]):
        if y_test[i][j] == 1:
            target = j
            break
    if hypothesis == target:
#        print("SUCCESS: Guessed " + str(hypothesis) + " correctly")
        successes += 1
    else:
#        print("FAILED: Guessed " + str(hypothesis) + " not " + str(target))
        failures += 1

final_result = successes*100/(successes + failures)
print("Final Grade: " + str(final_result) + "%")

#Input drawing

img = zeros([28,28])
h = len(img)
w = len(img[0])

for y in range(h):
  for x in range(w):
    img[y,x] = 255

# imsave("Result.jpg",imga)

cv2.namedWindow("image")
cv2.setMouseCallback("image", draw)

while(true):
  cv2.imshow("image", white_img)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break












