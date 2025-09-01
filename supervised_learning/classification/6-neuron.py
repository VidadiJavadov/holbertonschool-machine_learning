#!/usr/bin/env python3
"""0-neuron"""

import numpy as np


class Neuron:
    """class which defines neuron"""

    def __init__(self, nx):
        """initializing function"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for Weight"""
        return self.__W

    @property
    def b(self):
        """getter for bias"""
        return self.__b

    @property
    def A(self):
        """getter for Act"""
        return self.__A

    def forward_prop(self, X):
        """forward_prop"""
        Z = np.matmul(self.__W, X) + self.b
        self.__A = 1 / (1 + np.exp(-Z))  # sigmoid activation
        return self.__A

    def cost(self, Y, A):
        """cost function"""
        m = Y.shape[1]
        cost = -(1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """evaluation"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent"""

        m = X.shape[1]

        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """train"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
