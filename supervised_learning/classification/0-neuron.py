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

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
