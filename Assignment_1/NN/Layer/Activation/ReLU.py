import numpy as np
from NN.Layer.Layer import Layer


class ReLU(Layer):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, layer_inputs):
        self.layer_inputs = layer_inputs
        # self.layer_outputs = np.where(self.layer_inputs >= 0, self.layer_inputs, 0)
        self.layer_outputs = np.maximum(0, self.layer_inputs)

    def backward(self, grad_in):
        self.grad_out = grad_in.copy()
        self.grad_out[self.layer_inputs <= 0] = 0
