#!/usr/bin/env python3
"""one hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """one hot decode"""
    if isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    return np.argmax(one_hot)
