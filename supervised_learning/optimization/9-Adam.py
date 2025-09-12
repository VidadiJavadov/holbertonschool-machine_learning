#!/usr/bin/env python3
"""Adam optimizer"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """update variables with Adam"""
    v = beta1 * v + (1-beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_bias_corr = v / (1 - beta1**t)
    s_bias_corr = s / (1 - beta2**t)
    var = var - alpha * v_bias_corr / (np.sqrt(s_bias_corr) + epsilon)
