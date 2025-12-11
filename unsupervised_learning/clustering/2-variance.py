#!/usr/bin/env python3
"""Computes total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculates total intra-cluster variance."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(C, np.ndarray)
        or C.ndim != 2
    ):
        return None

    # Compute squared distances between each point and each centroid
    # (n, d) -> (n, 1, d)
    # (k, d) -> (1, k, d)
    # result -> (n, k, d)
    diff = X[:, None, :] - C[None, :, :]

    # Squared distances -> (n, k)
    dist_sq = (diff ** 2).sum(axis=2)

    # For each point, take distance to the closest centroid -> (n,)
    min_dist_sq = np.min(dist_sq, axis=1)

    # Total variance (scalar)
    return np.sum(min_dist_sq)
