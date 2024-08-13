
# MNIST Digit Recognition with TensorFlow

## Project Overview

This project implements a neural network model for recognizing handwritten digits using the MNIST dataset. It demonstrates the use of TensorFlow for building and training a simple neural network, as well as custom data parsing and visualization tools.

## Features

- Custom MNIST dataset parser
- TensorFlow-based neural network model
- Model training and evaluation
- Console-based image visualization
- Checkpointing for model persistence

## File Structure

```
.
├── README.md
├── data_parser.py
├── data_visualizer.py
├── log.py
├── main.py
├── checkpoints/
│   └── model.keras (will be created automatically once the script runs)
├── t10k-images.idx3-ubyte (must be downloaded by user)
├── t10k-labels.idx1-ubyte (must be downloaded by user)
├── train-images.idx3-ubyte (must be downloaded by user)
└── train-labels.idx1-ubyte (must be downloaded by user)
```

## Requirements

- Python 3.x
- TensorFlow
- NumPy

## Installation

1. Clone this repository:

2. Install the required packages:
   ```
   pip install tensorflow numpy
   ```

3. Download the MNIST dataset files and put them in the same directory as `main.py` (if not already included):
   - train-images.idx3-ubyte
   - train-labels.idx1-ubyte
   - t10k-images.idx3-ubyte
   - t10k-labels.idx1-ubyte

## Usage

Run the main script to train the model and evaluate it on test data:

```
python main.py
```

This will:
1. Parse the MNIST dataset
2. Train the neural network model (if no checkpoint exists)
3. Load a pre-trained model (if a checkpoint exists)
4. Evaluate the model on random samples from the test set

## Project Components

### data_parser.py
Contains custom functions for parsing the MNIST dataset files and converting them into NumPy arrays.

### data_visualizer.py
Provides functionality to visualize random images from the dataset in the console.

### log.py
Implements custom logging and progress tracking utilities.

### main.py
The main script that ties everything together, including model definition, training, and evaluation.

## Model Architecture

The neural network model consists of:
- Input layer: 784 neurons (28x28 flattened image)
- Hidden layer 1: 30 neurons with ReLU activation
- Hidden layer 2: 20 neurons with ReLU activation
- Output layer: 10 neurons (one for each digit) with linear activation

## Future Improvements

- Implement data augmentation techniques
- Experiment with different model architectures
- Add a graphical user interface for drawing and recognizing digits
- Incorporate more advanced TensorFlow features
