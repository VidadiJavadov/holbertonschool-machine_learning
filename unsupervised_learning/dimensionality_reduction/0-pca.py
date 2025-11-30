#!/usr/bin/env python3
import numpy as np


def pca(X, var=0.95):
    """Perform PCA on X while preserving var fraction of variance."""
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    eig_vals = (S ** 2) / (X.shape[0] - 1)
    eig_vecs = Vt.T

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    explained = eig_vals / np.sum(eig_vals)
    cumulative = np.cumsum(explained)

    nd = np.searchsorted(cumulative, var) + 1
    nd = min(nd, eig_vecs.shape[1])

    return eig_vecs[:, :nd]
