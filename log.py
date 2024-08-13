from dataclasses import dataclass
from datetime import datetime

import numpy as np

debug_level = 0

class MessagePrinter:
    @staticmethod
    def info(msg: str):
        if debug_level > 1:
            print("INFO " + MessagePrinter._get_current_time_string() + ": " + msg)

    @staticmethod
    def warning(msg: str):
        if debug_level > 0:
            print('\033[93m' + "WARN:" + MessagePrinter._get_current_time_string() + ": " + msg + '\033[0m')

    @staticmethod
    def error(msg: str):
        print('\033[91m' + "ERROR:" + MessagePrinter._get_current_time_string() + ": " + msg + '\033[0m')
        
    @staticmethod
    def set_debug_level(a: int):
        if not (a in (0,1,2)):
            raise ValueError("debug level must be either 0, 1, 2")
        global debug_level
        debug_level = a

    @staticmethod
    def print_image_on_console(mnist_image_set: np.ndarray, idx: int):
        """
        Print an MNIST image from a dataset to the console using ASCII characters.

        This function takes a specified index from the MNIST image set and 
        prints a visual representation of the image to the console. Pixels with 
        values greater than 100 are represented by a solid block, while lower 
        values are represented by a space.

        Args:
            mnist_image_set (np.ndarray): A NumPy array containing the MNIST image set.
            idx (int): The index of the image in the MNIST image set to be printed.

        Returns:
            None
        """
        for row in mnist_image_set[idx]:
            for col in row:
                if col > 100:
                    print("▓", end="")
                else:
                    print(" ", end="")
            print()

    @staticmethod
    def _get_current_time_string() -> str:
        currentTime = datetime.now()
        return f"({currentTime.hour:02d}:{currentTime.minute:02d}:{currentTime.second})"

@dataclass(slots=True)
class ProgressPrinter:
    GOAL: int
    GOAL_DIGIT_COUNT: int
    value: int
    DESCRIPTION: str

    def __init__(self, goal:int, initial_value:int, description:str = ""):
        self.GOAL = goal
        self.value = initial_value
        self.DESCRIPTION = description
        self.GOAL_DIGIT_COUNT = len(str(goal))
        self._redraw()

    def update(self, value:int):
        self.value = value
        self._redraw()

    def step(self):
        self.value += 1
        self._redraw()
    
    def _redraw(self):
        if debug_level > 1:
            progress_count = f"{str(self.value).zfill(self.GOAL_DIGIT_COUNT)} / {self.GOAL}"
            progress_ratio = self.value / self.GOAL
            if self.value < self.GOAL:
                bars = ("█" * int(progress_ratio * 50)) + ("░" * int((1-progress_ratio) * 50))
            else:
                bars = "█" * 50
            print(f"{self.DESCRIPTION} {progress_count} [{bars}] {progress_ratio * 100 : .2f}%", end="\r")

    @staticmethod
    def finalize():
        print()