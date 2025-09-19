#!/usr/bin/env python3
"""L2 regularized gradient descent"""
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """gradient descent with l2"""
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # derivative of cost wrt Z for softmax output

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(l - 1)]
        W = weights["W" + str(l)]

        # gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # update weights and biases
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        if l > 1:
            # propagate dZ backward using derivative of tanh
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - cache["A" + str(l - 1)] ** 2)
