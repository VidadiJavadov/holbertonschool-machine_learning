#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np

def l2_reg_cost(cost, lambtha, weights, L, m):
    """function that computes l2_reg"""
    l2_reg = 0
    for i in range (1, L+1):
        W = weights["W"+str(i)]
        l2_reg += cost + lambtha / (2*m) * np.square(W) 
    return l2_reg
