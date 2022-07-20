import numpy as np
from NN.Layer.Layer import Layer


class TanH(Layer):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, layer_input):
        self.layer_inputs = layer_input
        self.layer_outputs = np.tanh(layer_input)

    def backward(self, grad_in):
        self.grad_out = grad_in * (1 - self.layer_outputs ** 2)
