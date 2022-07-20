import numpy as np
from NN.Layer.Linear import Linear


class Nadam:

    def __init__(self, *, learning_rate=0.001, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer: Linear):

        if not hasattr(layer, 'mt_w'):
            layer.mt_w = np.zeros_like(layer.weights)
            layer.mt_b = np.zeros_like(layer.biases)
            layer.vt_w = np.zeros_like(layer.weights)
            layer.vt_b = np.zeros_like(layer.biases)

        # https://ruder.io/optimizing-gradient-descent/index.html#adam
        # Update momentum (m_t's) with current gradients
        layer.mt_w = self.beta_1 * layer.mt_w + (1. - self.beta_1) * layer.grad_w
        layer.mt_b = self.beta_1 * layer.mt_b + (1. - self.beta_1) * layer.grad_b

        # bias correction
        bias_corrected_mt_w = layer.mt_w / (1. - self.beta_1 ** (self.iterations + 1))
        bias_corrected_mt_b = layer.mt_b / (1. - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients (v_t's)
        layer.vt_w = self.beta_2 * layer.vt_w + (1 - self.beta_2) * layer.grad_w ** 2
        layer.vt_b = self.beta_2 * layer.vt_b + (1 - self.beta_2) * layer.grad_b ** 2

        bias_corrected_vt_w = layer.vt_w / (1. - self.beta_2 ** (self.iterations + 1))
        bias_corrected_vt_b = layer.vt_b / (1. - self.beta_2 ** (self.iterations + 1))

        prod_factor = (1. - self.beta_1) / (1. - self.beta_1 ** (self.iterations + 1))

        layer.weights += -self.learning_rate * (self.beta_1 * bias_corrected_mt_w + prod_factor * layer.grad_w) /\
                         np.sqrt(bias_corrected_vt_w + self.epsilon)

        layer.biases += -self.learning_rate * (self.beta_1 * bias_corrected_mt_b + prod_factor * layer.grad_b) /\
                        np.sqrt(bias_corrected_vt_b + self.epsilon)

        self.iterations += 1


