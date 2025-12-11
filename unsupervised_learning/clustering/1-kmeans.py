#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def initialize(X, k):
    """Initialize centroids uniformly."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(k, int)
        or k <= 0
    ):
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # np.random.uniform #1
    return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Perform K-means with ONLY TWO LOOPS allowed."""
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

    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    for _ in range(iterations):       # ← LOOP 1
        # Vectorized distance computation (NO LOOPS)
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

        # Assign clusters (NO LOOPS)
        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        for i in range(k):            # ← LOOP 2
            points = X[clss == i]

            if points.size == 0:
                # np.random.uniform #2
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = points.mean(axis=0)

        # If centroids do not change → STOP
        if np.array_equal(C_prev, C):
            return C, clss

    return C, clss
