import struct
import numpy as np
import log

def training_set_parser(images_file_path: str):
    with open(images_file_path, "rb") as images_file:
        raw_data = images_file.read()

        magic = struct.unpack(">i", raw_data[:4])
        if magic[0] != 0x803:
            raise ValueError("This file is not a MNIST image file.")

        image_count = struct.unpack(">i", raw_data[4:8])[0]
        row_count = struct.unpack(">i", raw_data[8:12])[0]
        col_count = struct.unpack(">i", raw_data[12:16])[0]

        log.info("Image count:" + str(image_count))
        log.info("Row count:" + str(row_count))
        log.info("Col count:" + str(col_count))

        X = np.empty(shape=(image_count, row_count, col_count), dtype=np.uint8) # Create an array of [images[ columns[ rows ] ] ]

        for image_idx in range(image_count):
            image_offset = 16 + image_idx * (row_count * col_count)
            for row_idx in range(row_count):
                row_offset = image_offset + (row_idx * col_count) 
                for col_idx in range(col_count):
                    col_offset = row_offset + col_idx
                    X[image_idx, row_idx, col_idx] = raw_data[col_offset]
        return X

log.set_debug_level(2)
X = training_set_parser("train-images.idx3-ubyte")

def print_image_on_console(mnist_image_set: np.ndarray, idx: int):
    for row in X[idx]:
        for col in row:
            if col > 100:
                print("▓", end="")
            else:
                print(" ", end="")
        print()

print_image_on_console(0) # 5
print_image_on_console(1) # 0
print_image_on_console(2) # 4
print_image_on_console(3) # 1