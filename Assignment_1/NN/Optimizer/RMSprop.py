import numpy as np
from NN.Layer.Linear import Linear


class RMSprop:

    def __init__(self, *, learning_rate=0.001, epsilon=1e-7, beta=0.5):
        self.learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta = beta

    def update_params(self, layer: Linear):

        if not hasattr(layer, 'weight_cache'):
            layer.vt_w = np.zeros_like(layer.weights)
            layer.vt_b = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        # v_t = beta * v_{t-1} + (1- beta) grad^2
        layer.vt_w = self.beta * layer.vt_w + (1 - self.beta) * layer.grad_w ** 2
        layer.vt_b = self.beta * layer.vt_b + (1 - self.beta) * layer.grad_b ** 2

        # update parameters
        layer.weights += -self.learning_rate * layer.grad_w / (np.sqrt(layer.vt_w + self.epsilon))
        layer.biases += -self.learning_rate * layer.grad_b / (np.sqrt(layer.vt_b + self.epsilon))

        self.iterations += 1
