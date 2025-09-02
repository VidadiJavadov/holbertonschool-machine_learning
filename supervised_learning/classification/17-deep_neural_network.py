#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network class for binary classification."""

    def __init__(self, nx, layers):
        """Class constructor."""
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Private attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # One loop for validation + initialization
        prev_nodes = nx
        for i, nodes in enumerate(layers, start=1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            # He et al. initialization for weights
            self.__weights[f"W{i}"] = (
                np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            )
            # Biases initialized with zeros
            self.__weights[f"b{i}"] = np.zeros((nodes, 1))

            prev_nodes = nodes

    @property
    def L(self):
        """Getter for number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary."""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary."""
        return self.__weights
