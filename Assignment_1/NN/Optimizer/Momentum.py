import numpy as np
from NN.Layer.Linear import Linear


class Momentum:
    def __init__(self, *, learning_rate=0.005, gamma=0.05):
        self.learning_rate = learning_rate
        self.gamma = gamma

    def update_params(self, layer: Linear):
        if not hasattr(layer, 'momentum_w'):
            layer.momentum_w = np.zeros_like(layer.weights)
            layer.momentum_b = np.zeros_like(layer.biases)

        # update_t = γ · update_{t−1} + η ∇w_t
        layer.momentum_w = self.gamma * layer.momentum_w + self.learning_rate * layer.grad_w
        layer.momentum_b = self.gamma * layer.momentum_b + self.learning_rate * layer.grad_b

        layer.weights += -layer.momentum_w
        layer.biases += -layer.momentum_b
