#!/usr/bin/env python3
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    # --------- Validation ----------
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape
    if k > n:
        return None, None, None

    # --------- Initialize priors ----------
    pi = np.full(k, 1 / k)

    # --------- Initialize means using K-means ----------
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # --------- Initialize covariance matrices ----------
    S = np.tile(np.eye(d), (k, 1, 1))

    return pi, m, S
