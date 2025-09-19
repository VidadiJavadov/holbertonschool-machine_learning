#!/usr/bin/env python3
"""Dropout Gradient Descent"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Dropout"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # derivative for softmax + cross-entropy

    for i in range(L, 0, -1):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]

        # Gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update parameters
        weights["W" + str(i)] = W - alpha * dW
        weights["b" + str(i)] = weights["b" + str(i)] - alpha * db

        if i > 1:  # backprop through hidden layers
            dA_prev = np.matmul(W.T, dZ)
            dA_prev = dA_prev * (1 - cache["A" + str(i - 1)] ** 2)  # tanh'

            # Apply dropout mask (scale back properly)
            D_prev = cache["D" + str(i - 1)]
            dA_prev = (dA_prev * D_prev) / keep_prob

            dZ = dA_prev
