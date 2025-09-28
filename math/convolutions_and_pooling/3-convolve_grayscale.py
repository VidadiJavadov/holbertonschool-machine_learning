#!/usr/bin/env python3
"""general grayscale convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """performs convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

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
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    out_h = ((h + 2 * ph - kh) // sh) + 1
    out_w = ((w + 2 * pw - kw) // sw) + 1

    output = np.zeros((m, out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            region = padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
