#!/usr/bin/env python3
"""one hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """one hot decode"""
    return np.argmax(one_hot)
