#!/usr/bin/env python3
"""Compute total intra-cluster variance."""

import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set."""
    # Validate inputs
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(C, np.ndarray)
        or C.ndim != 2
        or X.shape[1] != C.shape[1]    # d must match
        or C.shape[0] == 0             # cannot have 0 centroids
    ):
        return None

    # X shape: (n, d)
    # C shape: (k, d)

    # Compute (X - C) differences with broadcasting
    # diff shape: (n, k, d)
    diff = X[:, None, :] - C[None, :, :]

    # Square and sum across dimensions → squared distances (n, k)
    dist_sq = (diff ** 2).sum(axis=2)

    # For each point, take minimum squared distance → shape (n,)
    min_dist_sq = np.min(dist_sq, axis=1)

    # Total variance = sum of all min distances (scalar)
    return np.sum(min_dist_sq)
