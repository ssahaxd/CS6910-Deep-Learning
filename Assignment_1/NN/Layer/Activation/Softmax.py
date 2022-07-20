import numpy as np
from scipy.special import softmax
from NN.Layer.Layer import Layer


class Softmax(Layer):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"{self.__class__.__name__}()"

    def forward(self, layer_inputs):
        self.layer_inputs = layer_inputs

        # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
        # exp_values = np.exp(layer_inputs - np.max(layer_inputs, axis=1, keepdims=True))
        # self.layer_outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.layer_outputs = softmax(layer_inputs, axis=1)

    def backward(self, grad_in):
        # https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
        # jacobian_m = np.diag(grad_in)
        #
        # for i in range(len(jacobian_m)):
        #     for j in range(len(jacobian_m)):
        #         if i == j:
        #             jacobian_m[i][j] = grad_in[i] * (1-grad_in[i])
        #         else:
        #             jacobian_m[i][j] = - grad_in[i] * grad_in[j]
        #
        # self.grad_out = jacobian_m
        self.grad_out = np.zeros_like(grad_in)
        for index, (output, grad) in enumerate(zip(self.layer_outputs, grad_in)):
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            self.grad_out[index] = np.dot(jacobian_matrix, grad)
