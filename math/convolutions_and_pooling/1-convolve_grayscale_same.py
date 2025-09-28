#!/usr/bin/env python3
"""convolve grayscale (same)"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """convolve grayscale (same)"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    pad_h_top = kh // 2
    pad_h_bottom = kh - 1 - pad_h_top
    pad_w_left = kw // 2
    pad_w_right = kw - 1 - pad_w_left

    padded = np.pad(images,
                    ((0, 0), (pad_h_top, pad_h_bottom),
                             (pad_w_left, pad_w_right)),
                    mode='constant')
    
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
