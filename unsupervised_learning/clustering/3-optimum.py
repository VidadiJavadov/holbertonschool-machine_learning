#!/usr/bin/env python3
"""Optimum K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(kmin, int) or kmin <= 0 or
        (kmax is not None and
         (not isinstance(kmax, int) or kmax < kmin)) or
        not isinstance(iterations, int) or iterations <= 0):
        return None, None

    # If kmax not provided → use n
    if kmax is None:
        kmax = X.shape[0]

    # Must analyze at least 2 cluster sizes
    if kmax - kmin + 1 < 2:
        return None, None

    results = []
    vars_list = []

    # ----- LOOP 1: run kmeans for each k -----
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None
        vars_list.append(var)

    # ----- LOOP 2: compute delta variance -----
    base = vars_list[0]
    d_vars = [v - base for v in vars_list]

    return results, d_vars
