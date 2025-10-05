#!/usr/bin/env python3
"""Convolutional neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation over a convolutional layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant'
    )

    h_new = int((h_prev + 2 * pad_h - kh) / sh) + 1
    w_new = int((w_prev + 2 * pad_w - kw) / sw) + 1
    Z = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start, vert_end = h * sh, h * sh + kh
                    horiz_start, horiz_end = w * sw, w * sw + kw
                    A_slice = A_prev_pad[
                        i, vert_start:vert_end, horiz_start:horiz_end, :
                    ]
                    Z[i, h, w, c] = (
                        np.sum(A_slice * W[:, :, :, c]) + b[:, :, :, c]
                    )

    return activation(Z)
