#!/usr/bin/env python3
"""Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    rnn_cells: list of RNNCell instances (length l)
    X: input data of shape (t, m, i)
    h_0: initial hidden states of shape (l, m, h)

    Returns:
    H: hidden states for all layers and time steps
       shape: (t + 1, l, m, h)
    Y: outputs for all time steps
       shape: (t, m, o)
    """

    t, m, _ = X.shape
    l, _, h = h_0.shape

    # Get output size from last RNN cell
    o = rnn_cells[-1].Wy.shape[1]

    # Initialize hidden states tensor
    H = np.zeros((t + 1, l, m, h))

    # Set initial hidden states
    H[0] = h_0

    # Initialize outputs
    Y = np.zeros((t, m, o))

    # Forward propagation
    for step in range(t):

        x = X[step]

        for layer in range(l):

            h_prev = H[step, layer]

            # Input comes from previous layer (or X for first layer)
            if layer == 0:
                layer_input = x
            else:
                layer_input = H[step + 1, layer - 1]

            # Forward pass through RNN cell
            h_next, y = rnn_cells[layer].forward(h_prev, layer_input)

            # Store next hidden state
            H[step + 1, layer] = h_next

            # Output only from last layer
            if layer == l - 1:
                Y[step] = y

    return H, Y
