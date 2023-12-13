import numpy as np
from numpy.typing import NDArray

from .interface import ActivationFn


class ReLU(ActivationFn):
    def __init__(self):
        pass

    def evaluate(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        The Rectified Linear Unit Activation Function

        Args:
            inp (NDArray[np.float64]): the input to the sigmoid

        Returns:
            NDArray[np.float64]: relu applied to the input
        """
        return np.maximum(0, inp)

    def prime(self, inp: NDArray[np.float64]):
        """
        Derivative of the ReLU Activation Function

        NOTE: We take the derivative at x = 0 to be 0

        Args:
            inp (NDArray[np.float64]): the input

        Returns:
            NDArray[np.float64]: the derivative of relu function applied to the input
        """
        return np.where(inp <= 0, 0.0, 1.0)
