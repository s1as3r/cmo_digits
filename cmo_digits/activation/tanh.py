import numpy as np
from numpy.typing import NDArray

from .interface import ActivationFn


class Tanh(ActivationFn):
    def __init__(self):
        pass

    def evaluate(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Hyperbolic Tan Activation Function

        Args:
            inp (NDArray[np.float64]): the input to tanh

        Returns:
            NDArray[np.float64]: the tanh function applied to the input
        """
        return (np.exp(inp) - np.exp(-inp)) / (np.exp(inp) + np.exp(-inp))

    def prime(self, inp: NDArray[np.float64]):
        """
        Derivative of the tanh Activation Function

        Args:
            inp (NDArray[np.float64]): the input

        Returns:
            NDArray[np.float64]: the derivative of tanh function applied to the input
        """
        return 1 - self.evaluate(inp) ** 2
