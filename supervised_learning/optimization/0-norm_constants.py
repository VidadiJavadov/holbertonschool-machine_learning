#!/usr/bin/env python3
"norm constants"
import numpy as np


def normalization_constants(X):
    """function for normalize constants"""
    mean = np.mean(X)
    std = np.std(X)

    return mean, std