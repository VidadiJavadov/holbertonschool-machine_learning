#!/usr/bin/env python3
import numpy as np
"""initialize k-means"""


def initialize(X, k):
    """Initializes K-means centroids with a multivariate uniform distribution."""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
        not isinstance(k, int) or k <= 0):
        return None

    # Minimum and maximum values per dimension
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # Initialize centroids uniformly between min_vals and max_vals
    centroids = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))

    return centroids
