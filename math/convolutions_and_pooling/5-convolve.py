#!/usr/bin/env python3
"""convolve multi-kernel images"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs convolution on images with multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = max((int(np.ceil(((h - 1) * sh + kh - h) / 2))), 0)
        pw = max((int(np.ceil(((w - 1) * sw + kw - w) / 2))), 0)
    else:
        raise ValueError("padding must be 'same', 'valid', or a tuple")

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w, nc))

    for i in range(out_h):
        for j in range(out_w):
            for k in range(nc):
                region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k], 
                                            axis=(1, 2, 3))

    return output
