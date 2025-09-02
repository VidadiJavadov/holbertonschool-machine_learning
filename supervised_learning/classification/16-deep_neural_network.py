#!/usr/bin/env python3

"""Deep Neural Network"""

import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNet"""

    def __init__(self, nx, layers):
        """init function"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # 2. Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not (np.issubdtype(layers.dtype, np.integer) and np.all(layers > 0)):
            raise TypeError("layers must be a list of positive integers")

        # Number of layers
        self.L = len(layers)

        # Cache to hold forward prop values
        self.cache = {}

        # Weights and biases
        self.weights = {}

        # He initialization of weights
        for l in range(1, self.L + 1):
            # Previous layer size (nx for first layer, otherwise previous layer's nodes)
            prev = nx if l == 1 else layers[l - 1]

            # He et al. method: N(0, sqrt(2/prev))
            self.weights["W" + str(l)] = (
                np.random.randn(layers[l - 1], prev) * np.sqrt(2 / prev)
            )

            # Biases start as zeros
            self.weights["b" + str(l)] = np.zeros((layers[l - 1], 1))