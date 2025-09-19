#!/usr/bin/env python3
"""L2 regularized gradient descent"""
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y (np.ndarray): One-hot labels of shape (classes, m)
        weights (dict): Dictionary of weights and biases
        cache (dict): Dictionary of layer outputs
        alpha (float): Learning rate
        lambtha (float): L2 regularization parameter
        L (int): Number of layers

    Returns:
        None (weights and biases updated in place)
    """
    m = Y.shape[1]
    # Initialize dZ for last layer (softmax)
    A_prev = cache["A" + str(L - 1)]
    A_L = cache["A" + str(L)]
    
    dZ = A_L - Y  # derivative of softmax cross-entropy
    for l in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(l - 1)]
        W = weights["W" + str(l)]
        
        # Compute gradients with L2 regularization
        dW = (np.dot(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        
        # Update weights and biases
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db
        
        if l > 1:
            # Backpropagate through tanh activation
            A_prev = cache["A" + str(l - 1)]
            dZ = np.dot(W.T, dZ) * (1 - A_prev ** 2)
