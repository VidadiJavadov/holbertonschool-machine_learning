#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def initialize(X, k):
    """Simple uniform initialization."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(k, int)
        or k <= 0
    ):
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # uniform used ONCE here
    return np.random.uniform(min_vals, max_vals, (k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """Simplest possible K-means (only numpy)."""
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

    # --- initialize centroids ---
    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    for _ in range(iterations):     # 1st allowed loop
        # Compute distances from every point to every centroid
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

        # Assign each point to closest centroid
        clss = np.argmin(distances, axis=1)

        # Save previous centroids to check for change
        C_prev = C.copy()

        # Update centroids
        for i in range(k):          # 2nd allowed loop
            points = X[clss == i]

            if points.size == 0:
                # 2nd and final allowed uniform call
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = points.mean(axis=0)

        # Convergence check
        if np.array_equal(C_prev, C):
            return C, clss

    return C, clss
