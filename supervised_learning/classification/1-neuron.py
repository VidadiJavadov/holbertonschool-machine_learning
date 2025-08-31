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
