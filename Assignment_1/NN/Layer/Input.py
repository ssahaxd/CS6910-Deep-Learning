from NN.Layer.Layer import Layer


class Input(Layer):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"InputLayer()"

    def forward(self, layer_inputs):
        self.layer_inputs = layer_inputs
        self.layer_outputs = self.layer_inputs

    def backward(self, grad_in):
        pass
