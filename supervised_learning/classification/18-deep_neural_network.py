#!/usr/bin/env python3
"""Classification algorithm using Deep Neural Network (DNN class)."""
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0 or not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization
        for i in range(self.__L):
            if i == 0:
                self.__weights["W1"] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights[f"W{i+1}"] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2 / layers[i-1])
            self.__weights[f"b{i+1}"] = np.zeros((layers[i], 1))

    @property
    def cache(self):
        return self.__cache

    @property
    def L(self):
        return self.__L

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation for the deep neural network"""
        self.__cache["A0"] = X

        for l in range(1, self.__L + 1):
            W = self.__weights[f"W{l}"]
            b = self.__weights[f"b{l}"]
            A_prev = self.__cache[f"A{l-1}"]

            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))  # sigmoid activation

            self.__cache[f"A{l}"] = A

        return A, self.__cache
