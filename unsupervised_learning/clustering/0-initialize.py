#!/usr/bin/env python3
"""Initialize K-means centroids."""

import numpy as np


def initialize(X, k):
    """Initialize K-means centroids uniformly within data bounds."""
    if (
        not isinstance(X, np.ndarray)
        or X.ndim != 2
        or not isinstance(k, int)
        or k <= 0
    ):
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))

    return centroids
