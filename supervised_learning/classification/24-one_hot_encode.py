#!/usr/bin/env python3
"""one hot encode"""
import numpy as np

def one_hot_encode(Y, classes):
    """one hot encode"""
    m = Y.shape[0]
    ohe_mat = np.zeros((classes, m))
    ohe_mat[Y, np.arange(m)] = 1
    return ohe_mat
