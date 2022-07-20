from NN.Layer.Input import Input
from NN.Layer.Layer import Layer
import numpy as np
import wandb


class Model:

    def __init__(self):
        self.layers = []

        self.loss = None
        self.optimizer = None
        self.scoring = None

        self.input_layer = None
        self.num_layers = 0
        self.trainable_layers = []
        self.trainable_layers_index = []

        self.training_result = []


    def __str__(self):
        model_str = "Model(\n"
        for layer in self.layers:
            model_str = model_str + f"\t{str(layer)}\n"
        model_str += ")"

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, scoring=None):
        self.loss = loss
        self.optimizer = optimizer
        self.scoring = scoring

    def setup_connections(self):
        self.input_layer = Input()
        self.num_layers = len(self.layers)

        for i in range(self.num_layers):
            if i == 0:
                self.layers[i].previous = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < self.num_layers - 1:
                self.layers[i].previous = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].previous = self.layers[i-1]
                self.layers[i].next = self.loss

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                self.trainable_layers_index.append(i)

        self.loss.set_trainable_layers(trainable_layers=self.trainable_layers)


    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer: Layer
            layer.forward(layer.previous.layer_outputs)

        return layer.layer_outputs

    # def forward(self, X, layer_index=None):
        # If idx is set to some index start forward pass from that index
        # if layer_index is None:
        #     assert X is not None, "X is None"
        #     self.input_layer.forward(X)
        #     layer_index = 0
        #
        # i: int = 0
        # for i in range(layer_index, self.num_layers):
        #     self.layers[i].forward(self.layers[i].previous.layer_outputs)
        #
        # return self.layers[i].layer_outputs

    def backward(self, y_predicted, y_true):
        self.loss.backward(y_predicted, y_true)

        for layer in reversed(self.layers):
            layer: Layer
            layer.backward(layer.next.grad_out)

    # def backward(self, y_predicted, y_true, idx=None):
        # if idx is None:
        #     idx = -1
        # for i in range(len(self.layers)-1, idx, -1):
        #     self.layers[i].backward(self.layers[i].next.grad_out)

    def train(self, X, Y, *, epochs=1, batch_size=None, validation_data=None, print_mini_batch=False, print_every=10):

        if batch_size is None:
            batch_size = len(X)

        assert batch_size <= len(X), f"{batch_size=} > {len(X)=}"

        num_batches = len(X)//batch_size

        if num_batches * batch_size < len(X):
            num_batches += 1

        for epoch in range(1, epochs+1):
            print(f"{epoch=}")
            self.loss.reset()
            self.scoring.reset()

            for batch in range(num_batches):
                batch_X = X[batch*batch_size: (batch+1)*batch_size]
                batch_Y = Y[batch * batch_size: (batch + 1) * batch_size]

                output = self.forward(batch_X)

                batch_loss, regularization_loss = self.loss.forward(output, batch_Y)
                total_batch_loss = batch_loss + regularization_loss
                batch_loss = np.mean(batch_loss)
                total_batch_loss = np.mean(total_batch_loss)

                batch_accuracy = self.scoring.calculate(output, batch_Y)
                self.backward(output, batch_Y)

                if self.optimizer.__class__.__name__ == "NAG":
                    self.optimizer.update_params(self, batch_X, batch_Y)
                else:
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)

                if print_mini_batch:
                    if batch % print_every == 0 or batch == num_batches-1:
                        print(f"\t{batch=}, {batch_loss=:.2f}, {regularization_loss=:.2f}, {total_batch_loss=:.2f}, "
                              + f"{batch_accuracy= :.2f}")


            loss = self.loss.get_epoch_loss()
            accuracy = self.scoring.calculate_epoch_accuracy()

            print(f"training_{loss=:.2f}, training_{accuracy= :.2f}")
            epoch_results = {"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy}

            if validation_data is not None:
                X_val, Y_val = validation_data
                # self.validate(X_val, Y_val, batch_size=batch_size)
                validation_result = self.test(X_val, Y_val)

                epoch_results = {
                    **epoch_results,
                    "val_loss": validation_result['loss'],
                    "val_accuracy": validation_result["accuracy"]
                    }
                print(f"Validation: val_loss={validation_result['loss']:.2f}, "
                      f"val_accuracy={validation_result['accuracy'] :.2f}\n")
            try:
                wandb.log(epoch_results)
            except wandb.errors.Error:
                pass

    def test(self, X, Y, *, batch_size=None):
        self.scoring.reset()
        self.loss.reset()

        if batch_size is None:
            batch_size = len(X)

        assert batch_size <= len(X), f"{batch_size=} > {len(X)=}"

        num_batches = len(X) // batch_size
        if num_batches * batch_size < len(X):
            num_batches += 1

        for batch in range(num_batches):
            batch_X = X[batch * batch_size: (batch + 1) * batch_size]
            batch_Y = Y[batch * batch_size: (batch + 1) * batch_size]

            output = self.forward(batch_X)
            batch_loss, regularization_loss = self.loss.forward(output, batch_Y)
            total_batch_loss = batch_loss + regularization_loss

            self.scoring.calculate(output, batch_Y)

        loss = self.loss.get_epoch_loss()
        accuracy = self.scoring.calculate_epoch_accuracy()
        return {"loss": loss, "accuracy": accuracy}

    def predict(self, X, *, batch_size=None):

        if batch_size is None:
            batch_size = len(X)

        assert batch_size <= len(X), f"{batch_size=} > {len(X)=}"

        num_batches = len(X) // batch_size

        if num_batches * batch_size < len(X):
            num_batches += 1

        predictions = []
        for batch in range(num_batches):
            batch_X = X[batch * batch_size: (batch + 1) * batch_size]
            # outputs = self.forward(batch_X)
            # predictions.append(np.argmax(outputs))
            predictions.append(self.forward(batch_X))

        return np.vstack(predictions)
