#!/usr/bin/env python3
"""bibackward net"""
import numpy as np

class BidirectionalCell:
    def __init__(self, i, h, o):
        """
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, h_next, x_t):
        """
        h_next: numpy.ndarray of shape (m, h) containing the next hidden state
        x_t: numpy.ndarray of shape (m, i) containing the data input for the cell

        Returns: h_prev, the previous hidden state
        """
        m, h = h_next.shape
        # Concatenate x_t and h_next (as we do in a regular RNN)
        concat = np.concatenate((x_t, h_next), axis=1)  # shape (m, i + h)
        
        # Compute h_prev using tanh activation
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bh)  # shape (m, h)
        
        return h_prev
