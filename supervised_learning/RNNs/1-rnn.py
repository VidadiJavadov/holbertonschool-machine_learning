#!/usr/bin/env python3
"""RNN implementation"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN

    Arguments:
    rnn_cell -- instance of RNNCell
    X -- numpy array of shape (t, m, i)
    h_0 -- numpy array of shape (m, h)

    Returns:
    H -- hidden states (t+1, m, h)
    Y -- outputs (t, m, o)
    """

    # Get dimensions
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]

    # Initialize H and Y
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))

    # Set initial hidden state
    H[0] = h_0

    # Forward propagation through time
    for step in range(t):

        # Current input
        x_t = X[step]

        # Previous hidden state
        h_prev = H[step]

        # Run one RNN cell step
        h_next, y = rnn_cell.forward(h_prev, x_t)

        # Store results
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
