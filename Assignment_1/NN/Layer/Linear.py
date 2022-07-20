import numpy as np
from NN.Layer.Layer import Layer


class Linear(Layer):

    def __init__(self, n_inputs, n_neurons, *, initializer='random', alpha=0):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.alpha = alpha  # Regularization parameter

        self.weights = None
        self.biases = None
        self.grad_w = None
        self.grad_b = None


        # Initializing the weights
        if initializer == 'random':
            # self.weights = np.random.uniform(-1, 1, (n_inputs, n_neurons))
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        elif initializer == 'xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs)
            # limit = np.sqrt(6.0/(n_inputs+n_neurons))
            # self.weights = np.random.uniform(-limit, limit, (n_inputs, n_neurons))

        elif initializer == 'he':
            self.weights = np.random.randn(n_inputs, n_neurons) / np.sqrt(n_inputs / 2)

        # Initializing the biases
        # self.biases = np.zeros((1, n_neurons))
        self.biases = np.ones((1, n_neurons))
        # self.biases = np.random.uniform(-1, 1, (1, n_neurons))



    def __str__(self):
        return f"{__class__.__name__}({self.n_inputs}x{self.n_neurons})"


    def forward(self, layer_inputs):
        self.layer_inputs = layer_inputs
        self.layer_outputs = np.dot(layer_inputs, self.weights) + self.biases

    def backward(self, grad_in):
        self.grad_out = np.dot(grad_in, self.weights.T)
        self.grad_w = np.dot(self.layer_inputs.T, grad_in)
        self.grad_b = np.sum(grad_in, axis=0, keepdims=True)

        if self.alpha > 0:
            self.grad_w += self.alpha * self.weights
            self.grad_b += self.alpha * self.biases


