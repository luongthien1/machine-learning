from Layer import Layer
import numpy as np

class FullConnectedLayer(Layer) :
    def __init__(self, inputshape, outputshape):

        inputshape = inputshape
        outputshape= outputshape
        self.weight = np.random.rand(inputshape[1], outputshape[1]) - 0.5
    
    def forward_propagation(self, input: np.ndarray):
        self.input = input
        self.output = self.input.dot(self.weight)
        return self.output
    
    def backward_propagation(self, output_error: np.ndarray, learning_rate):
        currrent_layer_error = output_error.dot(self.weight.T)
        dweight = self.input.T.dot(output_error)
        self.weight -= learning_rate * dweight

        return currrent_layer_error
