#!/usr/bin/env python3
"norm constants"
import numpy as np


def normalization_constants(X):
    """function for normalize constants"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return mean, std