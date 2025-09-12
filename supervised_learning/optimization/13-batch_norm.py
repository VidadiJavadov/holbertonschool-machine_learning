#!/usr/bin/env python3
"""batch notmalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """batch norm"""
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    Z_hat = (Z - mean)/np.sqrt(var + epsilon)
    Z_norm = gamma * Z_hat + beta
    return Z_norm
