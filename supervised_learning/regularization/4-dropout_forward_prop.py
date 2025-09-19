#!/usr/bin/env python3
"""Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """dropout technique in forward prop"""
    cache = {}
    cache["A0"] = X

    for i in range(1, L+1):
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]
        A_prev = cache["A" + str(i-1)]

        Z = np.matmul(W, A_prev) + b

        if i<L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            cache["D" + str(i)] = D.astype(int)
            A = (A * D) / keep_prob
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True) 
        
        cache["A" + str(i)] = A

    return cache
