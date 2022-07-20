import numpy as np
from NN.Layer.Layer import Layer


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, layer_input):
        self.layer_inputs = layer_input
        self.layer_outputs = 1. / (1. + np.exp(-layer_input))

        # self.layer_outputs = np.where(layer_input >= 0,
        #                               1./(1. + np.exp(-layer_input)),
        #                               np.exp(layer_input) / (1. + np.exp(layer_input))
        #                               )

    def backward(self, grad_in):
        self.grad_out = grad_in * self.layer_outputs * (1 - self.layer_outputs)
