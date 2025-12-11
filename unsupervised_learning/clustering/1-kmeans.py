#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Perform K-means on X."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(k, int)
        or k <= 0
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        return None, None

    n, d = X.shape

    # Initialize centroids using 0-initialize.py (1st uniform call happens there)
    C = initialize(X, k)
    if C is None:
        return None, None

    # Precompute bounds for possible reinitialization of empty clusters
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    for _ in range(iterations):      # 1st loop
        # Compute distances from each point to each centroid: shape (n, k)
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        # Update centroids: mean of points in each cluster
        for i in range(k):          # 2nd loop
            points = X[clss == i]
            if points.size == 0:
                # 2nd and last use of np.random.uniform
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = points.mean(axis=0)

        # If centroids didn't change, we are done
        if np.array_equal(C_prev, C):
            return C, clss

    # If we exit by reaching max iterations, return last C and clss
    return C, clss
