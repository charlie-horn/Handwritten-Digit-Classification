# Machine-Learning

baseline.py : example taken from https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

			- Results in a %1.78 error
			- 10 Epochs
			- Updates every 200 images
			- One hidden layer
				- Rectifier activation layer
			- Softmax activation function on output layer
			- Logarithmic loss function (categorical_crossentropy)
			- ADAM gradient descent function for weighting
			- 60000 training images
			- 10000 test images

scratch.py : Neural network implementation to classify handwritten digits from 0-9
    
    - Dataset : MNIST digits from kaggle
        - 60000 training images
        - 10000 test images
    - Structure : 
        - Input layer :
            - P = Pixels = 784 (28x28 images)
        - Intermediate layer:
            - Currently D=1 layer 
                - TODO deepen neural net and evaluate optimal number of intermediate layers
                - See branch 'add_depth'
                - Biggest problem is finding the derivative of the Cost function wrt. the weight matrices
            - M: 784 neurons (TODO tweak sizes)
        - Output layer :
            - K: 10 neurons (number of classes)
    - Usage:
        - $ python scratch.py :
            - Train on 60000 training images
            - Draw digit with mouse, press 'q' when finished
            - Program will print its best guess
        - $ python scratch.py --test :
            - '-E' :
                - Number of epochs
                - Default = 10
            - '-T'
                - Test the configuration
                - test on 10000 labeled images, and return results as a percentage
            - '-L' :
                - Learning rate
                - Default = '0.000000001'
            - '-M' :
                - Size of the intermediate layer
                - Default = 784
    - Results :
        - M = 784, E = 100, L = 0.0000001 : %9.1 
        - M = 1000, E = 100, L = 0.0000001 : %15.08
