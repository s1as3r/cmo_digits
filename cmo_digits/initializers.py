from typing import Callable, List

import numpy as np
from numpy._typing import NDArray

WeightInitializer = Callable[[List[int]], List[NDArray[np.float64]]]


def he_kaiming(sizes: List[int]) -> List[NDArray[np.float64]]:
    """
    Kaiming Initialization

    the returned weights are zero-centered gaussians with standard deviation of sqrt(2/n_l) where n_l is the number of neurons in the previous/input layer.

    Args:
        sizes (List[int]): the sizes of the layers

    Returns:
        List(NDArray[np.float64]): the weights
    """
    return [
        np.random.randn(y, x) * np.sqrt(2.0 / x) for x, y in zip(sizes[:-1], sizes[1:])
    ]


def gaussian(sizes: List[int]) -> List[NDArray[np.float64]]:
    """
    initialie weigts and biases randomly with a gaussian distribution
    of mean 0 and variance 1

    Args:
        sizes (List[int]): the sizes of the layers

    Returns:
        List(NDArray[np.float64]): the weights
    """
    return [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
