import numpy as np

from NN.Layer.Linear import Linear


class MeanSquaredError:
    # def __init__(self):
    #     self.grad_out = None
    #
    # def __str__(self):
    #     return f"{__class__.__name__}()"
    #
    # @staticmethod
    # def forward(y_predicted, y_true):
    #     return np.mean((y_true - y_predicted) ** 2)
    #
    #
    # def backward(self, y_predicted, y_true):
    #     self.grad_out = -1 * (y_true - y_predicted)

    def __init__(self):
        self.grad_out = None
        self.batch_loss_sofar = 0
        self.batch_size_sum = 0
        self.trainable_layers = None

    def __str__(self):
        return f"{__class__.__name__}()"

    def set_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def forward(self, y_predicted, y_true):
        regularization_loss = 0
        for layer in self.trainable_layers:
            layer: Linear
            if layer.alpha > 0:
                regularization_loss += layer.alpha * np.sum(layer.weights * layer.weights)
                regularization_loss += layer.alpha * np.sum(layer.biases * layer.biases)

        clipped_y_predicted = np.clip(y_predicted, 1e-7, 1-1e-7)
        # # log_likelihood = -np.log(clipped_y_predicted)
        # # return np.sum(log_likelihood * y_true, axis=1)
        # batch_neg_log_loss = -np.log(np.sum(clipped_y_predicted * y_true, axis=1))
        batch_MSE_loss = np.mean((y_true - clipped_y_predicted) ** 2, axis=1)

        self.batch_size_sum += len(y_predicted)
        self.batch_loss_sofar += np.sum(batch_MSE_loss)

        return batch_MSE_loss, regularization_loss

    def backward(self, y_predicted, y_true):
        batch_size = len(y_predicted)
        num_classes = len(y_predicted[0])

        self.grad_out = -1 * (y_true - y_predicted) / num_classes
        self.grad_out = self.grad_out / batch_size



    def get_epoch_loss(self):
        epoch_loss = self.batch_loss_sofar/self.batch_size_sum
        self.reset()
        return epoch_loss

    def reset(self):
        self.batch_loss_sofar = 0
        self.batch_size_sum = 0
