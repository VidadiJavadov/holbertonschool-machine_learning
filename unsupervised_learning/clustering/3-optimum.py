#!/usr/bin/env python3
"""Finds the optimum number of clusters."""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance."""
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

    # Only set kmax when not provided
    if kmax is None:
        kmax = X.shape[0]

    # Must compare at least 2 different k
    if (kmax - kmin + 1) < 2:
        return None, None

    results = []
    variances = []

    for k in range(kmin, kmax + 1):   # LOOP 1
        C, clss = kmeans(X, k, iterations)
        if C is None:
            return None, None
        results.append((C, clss))

        var = variance(X, C)
        if var is None:
            return None, None
        variances.append(var)

    # differences
    base = variances[0]
    d_vars = [v - base for v in variances]

    return results, d_vars
