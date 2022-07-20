import numpy as np
from NN.Layer.Linear import Linear


class SGD:
    def __init__(self, *, learning_rate=0.005):
        self.learning_rate = learning_rate

    def update_params(self, layer: Linear):
        layer.weights += -self.learning_rate * layer.grad_w
        layer.biases += -self.learning_rate * layer.grad_b
