from random import randint

import numpy as np
import os
import data_parser
from log import MessagePrinter
MessagePrinter.set_debug_level(2)

MessagePrinter.info("Loading Tensorflow")
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
tf.autograph.set_verbosity(0)

CHECKPOINT_PATH = os.path.join(".", "checkpoints", "model.keras")

if (not os.path.exists(CHECKPOINT_PATH)):
    model = Sequential(
        [
            InputLayer((28*28,)),
            Dense(30, activation="relu"),
            Dense(20, activation="relu"),
            Dense(10, activation="linear"),
        ]
    )
    model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(learning_rate=0.001))
    train_X, train_Y = data_parser.MNIST_set_parser("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    train_X = train_X.reshape((-1, 28*28))

    MessagePrinter.info("Training started!")
    model.fit(train_X, train_Y, epochs=40)
    model.save(CHECKPOINT_PATH)
else:
    MessagePrinter.warning("Pretrained model is in use, found in " + str(CHECKPOINT_PATH))
    model = tf.keras.models.load_model(CHECKPOINT_PATH)

test_X, test_Y = data_parser.MNIST_set_parser("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")

for _ in range(10):
    i = randint(0, len(test_X) - 1)
    MessagePrinter.print_image_on_console(test_X, i)
    prediction = model.predict(test_X[i].reshape(1,28*28))
    MessagePrinter.info("Feeded a " + str(test_Y[i]))
    MessagePrinter.info("Model says its a " + str(np.argmax(prediction)))




