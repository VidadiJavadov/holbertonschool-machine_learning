#!/usr/bin/env python3
"""Compute total intra-cluster variance."""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(C, np.ndarray)
        or C.ndim != 2
        or X.shape[1] != C.shape[1]  # dimensions must match
        or C.shape[0] == 0           # need at least one centroid
    ):
        return None

    # X: (n, d), C: (k, d)
    # diff: (n, k, d)
    diff = X[:, None, :] - C[None, :, :]

    # squared distances: (n, k)
    dist_sq = (diff ** 2).sum(axis=2)

    # min squared distance for each point: (n,)
    min_dist_sq = np.min(dist_sq, axis=1)

    # total variance: scalar
    return np.sum(min_dist_sq)
