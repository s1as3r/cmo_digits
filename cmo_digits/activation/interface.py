from abc import abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ActivationFn(Protocol):
    """
    This is supposed to represent and Activation Function.
    We've made this an abstract base class with two functions
    `evaluate` and `prime`.
    """

    @abstractmethod
    def evaluate(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evaluate the Activation Function on the input

        Args:
            inp (NDArray[np.float64]): the input to the activation function

        Returns:
            NDArray[np.float64]: the ouput of the activation function
        """
        raise NotImplementedError

    @abstractmethod
    def prime(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Derivative of the Activation Function

        Args:
            inp (NDArray[np.float64]): the input

        Returns:
            NDArray[np.float64]: the derivative of the activation function applied to the input
        """
        raise NotImplementedError
