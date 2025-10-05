#!/usr/bin/env python3
"""Pooling forward propagation module."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation over a pooling layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h_new = int((h_prev - kh) / sh) + 1
    w_new = int((w_prev - kw) / sw) + 1
    A = np.zeros((m, h_new, w_new, c_prev))

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                vert_start, vert_end = h * sh, h * sh + kh
                horiz_start, horiz_end = w * sw, w * sw + kw
                A_slice = A_prev[i, vert_start:vert_end, 
                                 horiz_start:horiz_end, :]

                if mode == 'max':
                    A[i, h, w, :] = np.max(A_slice, axis=(0, 1))
                else:
                    A[i, h, w, :] = np.mean(A_slice, axis=(0, 1))

    return A
