import pickle
from typing import List, Self, Tuple

import numpy as np
from numpy.typing import NDArray

from .activation.interface import ActivationFn
from .activation.sigmoid import Sigmoid
from .initializers import WeightInitializer
from .initializers import gaussian as gaussian_init

TrainingData = List[Tuple[NDArray[np.float64], NDArray[np.float64]]]
TestingData = List[Tuple[NDArray[np.float64], int]]

DEFAULT_ACTIVATION_FN = Sigmoid


class Network:
    """
    Neural Network used to recognize handwritten digits.
    """

    def __init__(
        self,
        sizes: List[int],
        activation_fn: ActivationFn,
        weight_init_fn: WeightInitializer = gaussian_init,
    ):
        """
        Args:
            sizes (List[int]): number of neurons in each layer
                first layer is considered the input layer
                last layer is considered the output layer
                e.g: [2, 3, 1] means a network with an input layer of 2
                neurons, one hidden layer of three neurons and the output layer
                with one layer

            activation_fn (ActivationFn): the activation function
            activation_fn_prime (ActivationFn): the derivative of the activation function
            weight_init_fn (WeightInitializer): weight initialization function
        """
        self.num_layers = len(sizes)
        self.activation_fn = activation_fn

        self.sizes = sizes

        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = weight_init_fn(self.sizes)

        self.accuracy = []

    def feedforward(self, inp: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Feedforward to get output.

        Args:
            inp (NDArray[np.float64]): input array

        Returns:
            NDArray[np.float64]: output of the network for the input `inp`
        """
        for b, w in zip(self.biases, self.weights):
            inp = self.activation_fn.evaluate(np.dot(w, inp) + b)

        return inp

    def stochastic_gd(
        self,
        training_data: TrainingData,
        epochs: int,
        mini_batch_size: int,
        eta: float,
        test_data: TestingData | None = None,
    ):
        """
        The Stochastic Gradient Descent Optimization Algotithm

        Args:
            training_data (TrainingData): the training data, a list of tuples
                of inputs and their desired output

            epochs (int): number of epochs

            mini_batch_size (int): size of the mini-batch to use

            eta (float): learning rate

            test_data (TestingData | None): test data, a list of tuples
                of inputs and their desired output.
                tests would be run if provided. (default = None)
        """
        n_test = 0
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                evaluation = self.evaluate(test_data)
                self.accuracy.append(evaluation / n_test)
                print(f"Epoch {j}: {evaluation} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch: TrainingData, eta: float):
        """
        Update the networks weights and biases

        Args:
            mini_batch (TrainingData): a list of typles of inputs and the desired outputs

            eta (float): learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for inp, desired in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(inp, desired)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [
            w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(
        self, inp: NDArray[np.float64], desired: NDArray[np.float64]
    ) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
        """
        Backpropagation

        Args:
            inp (NDArray[np.float64]): input

            desired (NDArray[np.float64]): desired output activations

        Returns:
            Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
                nabla_w, nabla_b
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # forward pass
        activation = inp
        activations = [inp]  # activations layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activation_fn.evaluate(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(
            activations[-1], desired
        ) * self.activation_fn.prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = self.activation_fn.prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def predict(self, inp: NDArray[np.float64]) -> int:
        """
        get the predicted number for `inp`

        Args:
            inp (NDArray[np.float64]): the input image

        Returns:
            int: the predicted number
        """
        return int(self.feedforward(inp).argmax())

    def predict_with_score(self, inp: NDArray[np.float64]) -> Tuple[int, np.float64]:
        ff_res = self.feedforward(inp)
        return int(ff_res.argmax()), ff_res.max()

    def evaluate(self, test_data: TestingData) -> int:
        """
        Evaluate the network on `test_data`

        Args:
            test_data (TestingData): list of tuples of inputs and desired output

        Returns:
            int: the number of test inputs for which the nnet outputs the correct
            result
        """
        test_results = [(self.predict(x), y) for (x, y) in test_data]
        return sum(x == y for x, y in test_results)

    def cost_derivative(
        self, output_activations: NDArray[np.float64], desired: NDArray[np.float64]
    ):
        """
        Cost

        Args:
            output_activation (NDArray[np.float64]): the output layer activations

            desired (NDArray(np.float64)): the desired output layer activations
        """
        return output_activations - desired

    def save_to_pkl(self, filepath: str):
        """
        Save the network's weights and biases to `filepath` as a pickle

        Args:
            filepath (str): path to the pickle file
        """
        with open(filepath, "wb") as f:
            pickle.dump((self.weights, self.biases), f)

    @classmethod
    def from_pkl(cls, filepath: str) -> Self:
        """
        Load a network from a pickle

        Args:
            filepath (str): path to the pickle file

        Returns:
            Network: the neural network
        """
        with open(filepath, "rb") as f:
            weights, biases = pickle.load(f)

        sizes = [w.shape[1] for w in weights] + [weights[-1].shape[0]]
        net = cls(sizes, DEFAULT_ACTIVATION_FN())
        net.weights = weights
        net.biases = biases

        return net
