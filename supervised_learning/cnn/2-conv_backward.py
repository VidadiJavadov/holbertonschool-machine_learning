#!/usr/bin/env python3
"""backpropagation"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    # Padding for 'same'
    if padding == "same":
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
    else:
        pad_h, pad_w = 0, 0

    # Pad A_prev and dA_prev
    A_prev_pad = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )
    dA_prev_pad = np.zeros_like(A_prev_pad)

    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Backpropagation
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    # Slice from padded input
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Gradients
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]

        dA_prev_pad[i] = da_prev_pad

    # Remove padding
    if padding == "same":
        dA_prev = dA_prev_pad[:, pad_h:-pad_h or None, pad_w:-pad_w or None, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db
