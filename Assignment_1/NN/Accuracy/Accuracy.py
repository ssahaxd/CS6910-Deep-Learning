import numpy as np


class Accuracy:

    def __init__(self):
        self.batch_accuracy_sofar = 0
        self.batch_size_sum = 0

    def calculate(self, y_predicted, y_true):
        batch_size = len(y_predicted)

        results = (np.argmax(y_true, axis=1) == np.argmax(y_predicted, axis=1))

        self.batch_size_sum += batch_size
        self.batch_accuracy_sofar += np.sum(results)

        return np.mean(results)

    def calculate_epoch_accuracy(self):
        epoch_accuracy = self.batch_accuracy_sofar/self.batch_size_sum
        self.reset()
        return epoch_accuracy

    def reset(self):
        self.batch_accuracy_sofar = 0
        self.batch_size_sum = 0
