#!/usr/bin/env python3
"""0-neuron"""

class Neuron:
    """class which defines neuron"""

    def __init__(self, nx):
        """initializing function"""
        if not isinstance(self.nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")