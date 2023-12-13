import numpy as np
from numpy.typing import NDArray

from .interface import ActivationFn

DEFAULT_NEGATIVE_SLOPE = 0.01


class LReLU(ActivationFn):
    def __init__(self, negative_slope=DEFAULT_NEGATIVE_SLOPE):
        self.negative_slope = negative_slope

    def evaluate(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The Leaky Rectified Linear Unit Activation Function

        Args:
            inp (NDArray[np.float64]): the input to the sigmoid

        Returns:
            NDArray[np.float64]: relu applied to the input
        """
        return np.where(inp < 0, inp * self.negative_slope, inp)

    def prime(self, inp: NDArray[np.float64]):
        """
        Derivative of the Leaky-ReLU Activation Function

        NOTE: We take the derivative at x = 0 to be the negative_slope

        Args:
            inp (NDArray[np.float64]): the input

        Returns:
            NDArray[np.float64]: the derivative of relu function applied to the input
        """
        return np.where(inp <= 0, self.negative_slope, 1.0)
