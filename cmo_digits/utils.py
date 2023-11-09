from typing import Tuple

import numpy as np
import PIL.Image as im
from numpy.typing import NDArray

from .network import TestingData, TrainingData


def show_num(inp: NDArray[np.float64]):
    """
    show the number

    the image arrays that we use for training are 784x1 to make them
    easier to work with.

    Args:
        inp (NDArray[np.float64]): the image array
    """
    arr = (inp * 255).astype(np.uint8).reshape(28, 28)

    im.fromarray(arr, "L").show()


def load_data(filepath: str) -> Tuple[TrainingData, TestingData]:
    """
    load the training and testing data from `filepath`

    Args:
        filepath (str): the path to the dataset (a .npz file)

    Returns:
        Tuple[TrainingData, TestingData]: training and testing data
    """
    data = np.load(filepath)

    tr_d = (data["x_train"], data["y_train"])
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [_vectorize_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    te_d = (data["x_test"], data["y_test"])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return training_data, test_data


def _vectorize_result(j: int) -> NDArray[np.float64]:
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
