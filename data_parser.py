import struct
import numpy as np
from log import MessagePrinter, ProgressPrinter

def MNIST_set_parser(images_file_path: str, labels_file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse MNIST images and labels files, returning them as a tuple of NumPy arrays.

    This function reads the specified MNIST images and labels files, parses them, 
    and returns the images and labels as NumPy arrays. The function also checks 
    that the number of images matches the number of labels, raising an error if they do not.

    Args:
        images_file_path (str): The file path to the MNIST images file.
        labels_file_path (str): The file path to the MNIST labels file.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
            - The first array (X) contains the images from the MNIST dataset.
            - The second array (Y) contains the corresponding labels.

    Raises:
        ValueError: If the number of images does not match the number of labels.
        IOError: If there is an issue opening or reading either of the files.
    """
    X = _parse_images_file(images_file_path)
    Y = _parse_labels_file(labels_file_path)
    if len(X) != len(Y):
        MessagePrinter.error(f"Given files has {len(X)} images and {len(Y)} labels. Images and files are not matching.")
        raise ValueError("Images file and labels file has different amount of data.")
    return (X, Y)


def _parse_images_file(images_file_path: str) -> np.ndarray:
    """
    Parse an MNIST structured images file and return the images as a NumPy array.

    This function reads the specified file, which contains images from the MNIST 
    dataset in binary format, and returns them as a NumPy array. The function 
    first checks the file's magic number to verify its validity as an MNIST image file.

    Args:
        images_file_path (str): The file path to the MNIST images file.

    Returns:
        np.ndarray: A NumPy array containing the images from the MNIST dataset.
                    The shape of the array is (image_count, row_count, col_count).

    Raises:
        ValueError: If the file does not have the correct MNIST magic number.
        IndexError: If the file does not contain the claimed amount of data.
        IOError: If there is an issue opening or reading the file.
    """
    with open(images_file_path, "rb") as images_file:
        raw_data = images_file.read()

        # First 4 bytes are magic number (const 0x00000803)
        magic = struct.unpack(">i", raw_data[:4])
        if magic[0] != 0x803:
            MessagePrinter.error(f"{images_file_path} magic bytes are not correct.")
            raise ValueError("This file is not a MNIST image file.")

        MessagePrinter.info(f"{images_file_path} matches the magic number.")

        image_count = struct.unpack(">i", raw_data[4:8])[0]
        row_count = struct.unpack(">i", raw_data[8:12])[0]
        col_count = struct.unpack(">i", raw_data[12:16])[0]

        MessagePrinter.info("Image count:" + str(image_count))
        MessagePrinter.info("Row count:" + str(row_count))
        MessagePrinter.info("Col count:" + str(col_count))

        X = np.empty(shape=(image_count, row_count, col_count), dtype=np.uint8) # Create an array of [images[ columns[ rows ] ] ]

        try:
            progress_bar = ProgressPrinter(image_count, 0, "Load images")
            for image_idx in range(image_count):
                progress_bar.step()
                image_offset = 16 + image_idx * (row_count * col_count)
                for row_idx in range(row_count):
                    row_offset = image_offset + (row_idx * col_count) 
                    for col_idx in range(col_count):
                        col_offset = row_offset + col_idx
                        X[image_idx, row_idx, col_idx] = raw_data[col_offset]
            progress_bar.finalize()
        except IndexError as e:
            MessagePrinter.err(f"{images_file_path} file does not have claimed amount of data.\n" + str(e))
            raise
        return X

def _parse_labels_file(labels_file_path: str) -> np.ndarray:
    """
    Parse an MNIST structured labels file and return the labels as a NumPy array.

    This function reads the specified file, which contains the labels in the 
    MNIST dataset format, and returns them as a NumPy array.

    Args:
        labels_file_path (str): The file path to the MNIST labels file.

    Returns:
        np.ndarray: A NumPy array containing the labels from the MNIST dataset.

    Raises:
        IOError: If the file cannot be opened or read.
        ValueError: If the file contents cannot be interpreted as valid labels.
        IndexError: If the file length is smaller than its claimed size.
    """
    with open(labels_file_path, "rb") as label_file:
        raw_data = label_file.read()

        # First 4 bytes are magic number (const 0x00000801)
        magic = struct.unpack(">i", raw_data[:4])
        if magic[0] != 0x801:
            MessagePrinter.error(f"{labels_file_path} magic bytes are not correct.")
            raise ValueError("This file is not a MNIST image file.")
        
        label_count = struct.unpack(">i", raw_data[4:8])[0]
        MessagePrinter.info("Label count:" + str(label_count))

        Y = np.empty(label_count, dtype=np.uint8)

        try:
            progress_bar = ProgressPrinter(label_count, 0, "Load labels")
            for label_index in range(label_count):
                offset = 8 + label_index
                Y[label_index] = raw_data[offset]
                progress_bar.step()
            progress_bar.finalize()
        except IndexError as e:
            MessagePrinter.error(f"{labels_file_path} file does not have claimed amount of data.\n" + str(e))
            raise

        return Y