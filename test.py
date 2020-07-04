"""
This sample script trains a basic CNN on the
Fashion-MNIST dataset. It takes black and white images of clothing
and labels them as "pants", "belt", etc. This script is designed
to demonstrate the MLFlow integration with Keras.
"""
import random
from argparse import Namespace

# Import mlflow libraries
import mlflow
import mlflow.keras
from keras.callbacks import TensorBoard
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

mlflow.set_experiment("Default")


# Set hyperparameters, which can be overwritten with a W&B Sweep
hyperparameter_defaults = Namespace(
    dropout=0.2,
    hidden_layer_size=128,
    layer_1_size=16,
    layer_2_size=32,
    learn_rate=0.01,
    decay=1e-6,
    momentum=0.9,
    epochs=8,
)


config = hyperparameter_defaults

(X_train_orig, y_train_orig), (X_test, y_test) = fashion_mnist.load_data()

# Reducing the dataset size to 10,000 examples for faster train time
true = list(map(lambda x: True if random.random() < 0.167 else False, range(60000)))
ind = []
for i, x in enumerate(true):
    if x == True:
        ind.append(i)

X_train = X_train_orig[ind, :, :]
y_train = y_train_orig[ind]

img_width = 28
img_height = 28
labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

X_train = X_train.astype("float32")
X_train /= 255.0
X_test = X_test.astype("float32")
X_test /= 255.0

# reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(
    lr=config.learn_rate, decay=config.decay, momentum=config.momentum, nesterov=True
)

# build model
model = Sequential()
model.add(
    Conv2D(
        config.layer_1_size,
        (5, 5),
        activation="relu",
        input_shape=(img_width, img_height, 1),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(config.layer_2_size, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))
model.add(Flatten())
model.add(Dense(config.hidden_layer_size, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Add MLFLow Autolog
mlflow.keras.autolog()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=config.epochs)
