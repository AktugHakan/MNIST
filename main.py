from random import randint
import data_parser
from log import MessagePrinter

MessagePrinter.set_debug_level(2)

train_X, train_Y = data_parser.MNIST_set_parser("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
for _ in range(5):
    i = randint(0, len(train_X) - 1)
    MessagePrinter.info("Here comes a " + str(train_Y[i]))
    MessagePrinter.print_image_on_console(train_X, i)
