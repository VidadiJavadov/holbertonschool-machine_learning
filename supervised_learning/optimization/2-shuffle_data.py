#!/usr/bin/env python3
"""shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """function for shuffle the data"""
    X_shuffled = np.random.permutation(X)
    Y_shuffled = np.random.permutation(Y)

    return X_shuffled, Y_shuffled
