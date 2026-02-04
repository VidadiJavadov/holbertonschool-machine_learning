#!/usr/bin/env python3
"""bifoward net"""
import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell
    """

    def __init__(self, i, h, o):
        """
        Class constructor

        i: dimensionality of input
        h: dimensionality of hidden state
        o: dimensionality of output
        """

        # Forward direction weights and bias
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        # Backward direction weights and bias
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        # Output weights and bias
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step (forward direction)

        h_prev: previous hidden state (m, h)
        x_t: input at time t (m, i)

        Returns:
        h_next: next hidden state (m, h)
        """

        # Concatenate previous hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next
