#!/usr/bin/env python3
"""Finds the optimum number of clusters."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
    # -------- Input validation --------
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(kmin, int)
        or kmin <= 0
        or (kmax is not None and (not isinstance(kmax, int) or kmax < kmin))
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        return None, None

    if kmax is None:
        kmax = X.shape[0]

    if kmax - kmin + 1 < 2:
        # Must analyze at least 2 cluster sizes
        return None, None

    results = []
    variances = []

    # -------- LOOP #1: iterate over number of clusters --------
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None

        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None

        variances.append(var)

    # -------- Compute differences (NO LOOP needed) --------
    base = variances[0]               # smallest k variance
    d_vars = [v - base for v in variances]   # list comprehension allowed

    return results, d_vars
