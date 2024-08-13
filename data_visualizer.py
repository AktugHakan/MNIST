from random import randint

import numpy as np
from log import MessagePrinter


def show_random_images(X:np.ndarray, Y:np.ndarray, number_of_images:int):
    """
    Display a random set of example images and their labels from the dataset.

    This function randomly selects a specified number of images and their 
    corresponding labels from the given dataset and prints them to the console. 
    Each image is displayed using ASCII characters, and its label is shown 
    above the image.

    Args:
        X (np.ndarray): A NumPy array containing the images to display.
        Y (np.ndarray): A NumPy array containing the labels corresponding to the images.
        number_of_images (int): The number of random images to display.

    Returns:
        None
    """
    for _ in range(number_of_images):
        i = randint(0, len(X) - 1)
        MessagePrinter.info("Here comes a " + str(Y[i]))
        MessagePrinter.print_image_on_console(X, i)