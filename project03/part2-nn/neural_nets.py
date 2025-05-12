import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    return max(0,x)

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    return 1.0 if (x > 0.0) else 0.0

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1.0

class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """
    def __init__(self):
        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        # w  (3x2)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        # v  (3x )
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        # b     (3x )
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = 0.001
        self.epochs_to_train = 10
        #self.training_points = [((2,1), 10), ((3,3), 21), ((4,5), 32), ((6, 6), 42)]
        #self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10)]
        self.training_points = [((3, -4), -1), ((-5, 8), 3), ((4, 4), 8), ((6, 9), 15), ((-1, 6), 5), ((6, -8), -2),
                                ((-7, -1), -8), ((7, 1), 8), ((-4, -7), -11), ((-6, 6), 0)]

        self.testing_points = [(1,1), (2,2), (3,3), (5,5), (10,10),(4,4), (6,6), (7,7), (8,8), (9,9)]

    def train(self, x1, x2, y):
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights, input_values) + self.biases # (3 by 1 matrix)
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input) # TODO (3 by 1 matrix)

        output = np.dot(self.hidden_to_output_weights, hidden_layer_activation) # TODO
        activated_output = np.vectorize(output_layer_activation)(output) # TODO

        ### Backpropagation ###

        # Compute gradients
        # dC / df(u1) (1 by 1 matrix)
        output_layer_error = - (y - activated_output) # * np.vectorize(output_layer_activation_derivative)(self.hidden_to_output_weights)
        # TODO (3 by 1 matrix)
        hidden_layer_error = np.multiply( np.vectorize(output_layer_activation_derivative)(activated_output), self.hidden_to_output_weights.T ) * output_layer_error

        # TODO dC/db = dC/df * df /a (3 by 1 matrix)
        bias_gradients = np.multiply( hidden_layer_error, np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input) )
        # (3 by 1 matrix)
        hidden_to_output_weight_gradients = np.multiply(output_layer_error, hidden_layer_activation).transpose()
        input_to_hidden_weight_gradients = hidden_layer_error @ input_values.T # TODO

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate * hidden_to_output_weight_gradients
        '''
        
        ### Forward propagation ###
        # x     (:x3)
        input_values = np.matrix([[x1], [x2]])  # 2 by 1
        hidden_layer_weighted_input = np.dot(self.input_to_hidden_weights, input_values) + self.biases
        # Activation ReLU   (3x1 matrix)
        # f(z) = ReLU(z)
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)
        #    u = v.f(z)
        output = np.dot(self.hidden_to_output_weights, hidden_layer_activation)
        # f(u) = u
        activated_output = np.vectorize(output_layer_activation)(output)

        ### Backpropagation ###

        # Compute gradients
        #  dC/df = d[0.5(y-f(u))^2]/df = -(y-f(u))
        #  dC/df = -(y-f(u))
        output_layer_error = -(y - activated_output)
        hidden_layer_error = np.multiply(np.vectorize(output_layer_activation_derivative)(hidden_layer_weighted_input),
                                         self.hidden_to_output_weights.transpose()) * output_layer_error
        ### Gradient Descent ###
        #  (dz/db)(dReLU(z)/dz)*(du/df(z))*(df(u)/du)*(-y-f(u))
        bias_gradients = np.multiply(hidden_layer_error,
                                     np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input))
        bias_gradients = hidden_layer_error

        #bias_gradients = hidden_layer_error
        # [ (dC/df) * f(w.x + b) ]^T
        hidden_to_output_weight_gradients = np.multiply(output_layer_error, hidden_layer_activation).transpose()
        #  w(dC/df)*(df/dz).x^T
        input_to_hidden_weight_gradients = hidden_layer_error @ input_values.T
        # Use gradients to adjust weights and biases using gradient descent
        # b - n [(dC/df) * f(w.x + b)]^T
        # w = w - n *w(dC/df)*(df/dz).x^T
        # w1 = w1 - n[(dC/df) * f(w.x + b)]^T
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate * hidden_to_output_weight_gradients
        '''

    def predict(self, x1, x2):
        input_values = np.matrix([[x1],[x2]])
        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = self.input_to_hidden_weights @ input_values + self.biases # TODO
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input) # TODO
        output = self.hidden_to_output_weights @ hidden_layer_activation # TODO
        activated_output = output_layer_activation(output) # TODO
        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):
        #for epoch in range(self.epochs_to_train):
        for epoch in range(10):
            for x,y in self.training_points:
                #print(x,y)
                self.train(x[0], x[1], y)
                #print(self.input_to_hidden_weights)
            print("  EPOCH  ", epoch)
            print("(Input --> Hidden Layer) Weights:  ",self.input_to_hidden_weights)
            print("(Hidden --> Output Layer) Weights:  ",self.hidden_to_output_weights)
            print("Biases:  ",self.biases)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):
        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()
print("training Pairs", x.training_points)
print("(Input --> Hidden Layer) Weights: ",x.input_to_hidden_weights)
print("(Hidden --> Output Layer) Weights: ",x.hidden_to_output_weights)
print("Biases: ",x.biases)
x.train_neural_network()


# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
