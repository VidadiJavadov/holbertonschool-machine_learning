#!/usr/bin/env python3
"""K-means clustering."""

import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(k, int)
        or k <= 0
    ):
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # first uniform()
    C = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
    return C


def kmeans(X, k, iterations=1000):
    """Performs K-means clustering."""
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

    for _ in range(iterations):   # LOOP 1
        # distance matrix (NO LOOP)
        diff = X[:, :, None] - C.T[None, :, :]
        distances = np.sqrt((diff ** 2).sum(axis=1))

        clss = np.argmin(distances, axis=1)

        C_prev = C.copy()

        for i in range(k):        # LOOP 2
            points = X[clss == i]

            if points.size == 0:
                # second uniform()
                C[i] = np.random.uniform(min_vals, max_vals, (1, d))
            else:
                C[i] = points.mean(axis=0)

        if np.array_equal(C_prev, C):
            return C, clss

    return C, clss
