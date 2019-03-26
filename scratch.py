import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import math
import cv2
import sys

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

def forward(X, W):
  Z = sigmoid(X.dot(W1))
  Y = softmax(Z.dot(W2))
  return Y,Z

def last_grad(Z,T,Y):
  return Z.T.dot(Y - T)

def grad(X,Z,T,Y,W2):
  return X.T.dot(((Y - T).dot(W2.T)*(Z*(1 - Z))))

def draw(event, x, y, flags, param):
  global img, drawing
  if event == cv2.EVENT_LBUTTONDOWN:
    drawing = True
    img[y,x] = 0
  elif event == cv2.EVENT_MOUSEMOVE:
    if drawing:
      img[y,x] = 0
  elif event == cv2.EVENT_LBUTTONUP:
    drawing = False

# Parse command line args
for i,arg in enumerate(sys.argv):
    if arg == "-D":
        D = int(sys.argv[i+1])
        print("D: " + str(D))
    elif arg == "-M":
        M = [int(x) for x in sys.argv[i+1].split(',')]
        print("M: " + str(M))

if len(M) != D:
    sys.exit()

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
# P is the size of a sample
P = num_pixels
K = num_classes
#M = P

# Initialize Weights
W = []
for i in range(D + 1):
    if i == 0:
        W[i] = np.random.randn(P,M[i])
    elif i == D:
        W[i] = np.random.randn(M[i-1],K)
    else:
        W[i] = np.random.randn(M[i-1],M[i])

#Training

epochs = 10
learning_rate = 0.0000000001
C = 0

#while(not math.isnan(C)):
Z = []
gradient = []
for i in range(epochs):
    # Calculate internal layers
    for j in range(D + 1):
        if j == 0:
            Z[j] = sigmoid(x_train.dot(W[j]))
        elif j == D:
            Z[j] = softmax(Z[j-1].dot(W[j]))
        else:
            Z[j] = sigmoid(Z[j-1].dot(W[j]))
    # Calculate gradients wrt weights
    for j in range(D + 1):
        if j == D:
            gradient[j] = last_grad(Z[j-1], y_train, Z[-1])
        else:
            gradient[j] = grad(x_train,Z[j], y_train, Z[-1], W[j+1])
    # Calculate new weights
    for j in range(D + 1):
    #W2 -= learning_rate*grad_W2(Z, y_train, Y)
    #W1 -= learning_rate*grad_W1(x_train, Z, y_train, Y, W2)
        W[j] -= learning_rate*grad[j]
    C = cost(y_train, Z[-1])
    print(C)
#    learning_rate = learning_rate*10
#    print("New rate : " + str(learning_rate))

try:
    opt = sys.argv[1]
except:
    opt = None

#Testing
if opt == "--test":
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

else: 
    #Input drawing

    img = np.zeros([28,28])
    h = len(img)
    w = len(img[0])

    for y in range(h):
      for x in range(w):
        img[y,x] = 255

    # imsave("Result.jpg",imga)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw)
    drawing = False

    while(True):
      cv2.imshow("image", img)
      key = cv2.waitKey(1) & 0xFF
      if key == ord("q"):
        break

    # Test the image
    img = img.reshape(1, num_pixels).astype('float32') / 255

    Y, Z = forward(img, W1, W2)
    hypothesis = 0
    for i in range(Y.shape[1]):
        print(Y[0][i])
        if Y[0][i] > Y[0][hypothesis]:
            hypothesis = i

    print("Guess: " + str(hypothesis))

    cv2.destroyAllWindows()
