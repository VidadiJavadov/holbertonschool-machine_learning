#!/usr/bin/env python3
import numpy as np
"""initialize k-means"""

def initialize(X, k):
    """Initializes cluster centroids for K-means using multivariate uniform distribution."""
    try:
        # Validate input
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None
        if not isinstance(k, int) or k <= 0:
            return None

        # Compute min and max for each dimension: shape (d,)
        min_vals = X.min(axis=0)
        max_vals = X.max(axis=0)

        # Generate centroids using uniform distribution
        # numpy.random.uniform is used EXACTLY once
        centroids = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))

        return centroids

    except Exception:
        return None
