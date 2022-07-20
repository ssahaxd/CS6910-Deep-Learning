import numpy as np

from NN.Model import Model
from NN.Layer.Linear import Linear
from NN.Layer.Activation.TanH import TanH
from NN.Layer.Activation.Sigmoid import Sigmoid
from NN.Layer.Activation.Softmax import Softmax
from NN.Loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from NN.Optimizer.SGD import SGD
from NN.Accuracy.Accuracy import Accuracy
from NN.Layer.Activation.ReLU import ReLU
from NN.Loss.MeanSquaredError import MeanSquaredError
from NN.Optimizer.Adam import Adam
from NN.Optimizer.Nadam import Nadam
from NN.Optimizer.NAG import NAG
from NN.Optimizer.Momentum import Momentum
from NN.Optimizer.RMSprop import RMSprop
import wandb

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from keras.datasets import fashion_mnist
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0, stratify=y_train)


def do_standard_transform(X, scalar=None):
    X = X.reshape(X.shape[0], -1).astype("float32")

    # Or do sklearn's StandardScaler
    if scalar is not None:
        X = scalar.transform(X)
    else:
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    return X


def do_one_hot_encode(y, encoder=None):
    y = y.reshape(-1, 1)
    if encoder is not None:
        y = encoder.transform(y).toarray()
    else:
        encoder = OneHotEncoder()
        encoder.fit(y)
        y = encoder.transform(y).toarray()
    return y


X = do_standard_transform(x_train)
X_val = do_standard_transform(x_val)
X_test = do_standard_transform(x_test)

# One hot encoding the labels
enc = OneHotEncoder()
enc.fit(y_train.reshape(-1, 1))

y_train_one_hot = do_one_hot_encode(y_train, enc)
y_val_one_hot = do_one_hot_encode(y_val, enc)
y_test_one_hot = do_one_hot_encode(y_test, enc)

config_defaults = dict(
    epochs=5,
    num_layers=4,
    num_hidden_neurons=128,
    weight_decay=0,
    learning_rate=0.0001,
    optimizer="Adam",
    batch_size=16,
    activation="sigmoid",
    initializer="xavier",
    loss_function="MSE"
)

wandb.init(config=config_defaults, project="Assignment1", entity="cookie303", tags=["CE VS MSE"])

# Instantiate the model

model = Model()

# Add layers
model.add(Linear(X.shape[1], 128, initializer='xavier', ))
model.add(Sigmoid())
model.add(Linear(128, 128, initializer='xavier', ))
model.add(Sigmoid())
model.add(Linear(128, 128, initializer='xavier', ))
model.add(Sigmoid())
model.add(Linear(128, 10, initializer='xavier', ))
model.add(Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.0001),
    scoring=Accuracy()
)
model.setup_connections()

model.train(X, y_train_one_hot, epochs=5, batch_size=16, validation_data=(X_val, y_val_one_hot), print_mini_batch=True,
            print_every=500)
# model.backward(data, y_train_one_hot[:10])
# model.backward(data, np.array([[1,0,0],[1,0,0],[0,1,0]]))
# print(model.layers[0].grad_out)

# # Train the model
# model.train(X, y_train, validation_data=(X_test, y_test),
#             epochs=1, batch_size=32, print_every=421)


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#
# val_results = model.test(X_val, y_val_one_hot)
# prediction_probs = model.predict(X_val)
# predictions = np.argmax(prediction_probs, axis=1)
# ground_truth = np.argmax(y_val_one_hot, axis=1)
#
# wandb.log({"val_conf_mat": wandb.plot.confusion_matrix(probs=None,
#                                                        y_true=ground_truth, preds=predictions,
#                                                        class_names=class_names),
#            "test_loss": val_results['loss'],
#            "test_accuracy": val_results['accuracy']
#            })
#
test_results = model.test(X_test, y_test_one_hot)
prediction_probs = model.predict(X_test)
predictions = np.argmax(prediction_probs, axis=1)
ground_truth = np.argmax(y_test_one_hot, axis=1)

wandb.log({"test_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                        y_true=ground_truth, preds=predictions,
                                                        class_names=class_names),
           "test_loss": test_results['loss'],
           "test_accuracy": test_results['accuracy'],
           })

# # vibrant-dew-991
