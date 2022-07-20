from abc import ABC, abstractmethod


class Layer(ABC):

    def __init__(self):
        self.grad_out = None
        self.layer_outputs = None
        self.layer_inputs = None

        self.next: Layer = None
        self.previous: Layer = None

    def __str__(self):
        return f"{__class__.__name__}()"

    @abstractmethod
    def forward(self, layer_inputs):
        pass

    @abstractmethod
    def backward(self, grad_in):
        pass


