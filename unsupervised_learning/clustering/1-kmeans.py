#!/usr/bin/env python3
"""K-means clustering algorithm."""

import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering on dataset X."""
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

    # --- 1st use of uniform() → initialize centroids ---
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    C = np.random.uniform(min_vals, max_vals, (k, d))

    for _ in range(iterations):  # 1st allowed loop
        # Compute distances: shape (n, k)
        distances = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

        # Assign each point to the closest centroid
        clss = np.argmin(distances, axis=1)

        # Store previous centroids to check convergence
        C_prev = C.copy()

        # Update centroids (2nd allowed loop)
        for i in range(k):
            points = X[clss == i]
            if points.size == 0:
                # --- 2nd use of uniform() → reinitialize empty cluster ---
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = points.mean(axis=0)

        # Check if centroids stopped changing
        if np.allclose(C_prev, C):
            break

    return C, clss
