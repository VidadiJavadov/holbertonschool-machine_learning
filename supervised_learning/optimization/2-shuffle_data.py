#!/usr/bin/env python3
"""shuffle data"""
import numpy as np


def shuffle_data(X, Y):
    """function for shuffle the data"""
    m = X.shape[0]
    per = np.random.permutation(m)
    return X[per], Y[per]
