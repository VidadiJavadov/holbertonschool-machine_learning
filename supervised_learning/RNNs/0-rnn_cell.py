#!/usr/bin/env python3
"""RNN cell"""
import numpy as np


class RNNCell:
    """
    Represents a simple RNN cell
    """

    def __init__(self, i, h, o):
        """
        Initialize the RNN cell

        i: input dimension
        h: hidden dimension
        o: output dimension
        """

        # Weights for hidden state
        self.Wh = np.random.randn(i + h, h)

        # Bias for hidden state
        self.bh = np.zeros((1, h))

        # Weights for output
        self.Wy = np.random.randn(h, o)

        # Bias for output
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step

        h_prev: (m, h)
        x_t:    (m, i)

        Returns:
        h_next, y
        """

        # Concatenate hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Output (before softmax)
        z = np.matmul(h_next, self.Wy) + self.by

        # Softmax output
        y = self.softmax(z)

        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Compute softmax activation
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
