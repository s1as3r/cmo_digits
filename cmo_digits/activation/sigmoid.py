import numpy as np
from numpy.typing import NDArray

from .interface import ActivationFn


class Sigmoid(ActivationFn):
    @staticmethod
    def evaluate(inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Sigmoid Activation Function

        Args:
            inp (NDArray[np.float64]): the input to the sigmoid

        Returns:
            NDArray[np.float64]: the sigmoid function applied to the input
        """
        return 1.0 / (1.0 + np.exp(-inp))

    @staticmethod
    def prime(inp: NDArray[np.float64]):
        """
        Derivative of the Sigmoid Activation Function

        Args:
            inp (NDArray[np.float64]): the input

        Returns:
            NDArray[np.float64]: the derivative of sigmoid function applied to the input
        """
        return Sigmoid.evaluate(inp) * (1 - Sigmoid.evaluate(inp))
