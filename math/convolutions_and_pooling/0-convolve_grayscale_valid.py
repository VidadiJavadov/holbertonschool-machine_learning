#!/usr/bin/env python3
"""convolve grayscale"""
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """function for convolving"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    out_h = h - kh + 1
    out_w = w - kw + 1
    output = np.zeros((m, out_h, out_w))

    for i in out_h:
        for j in out_w:
            region = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
