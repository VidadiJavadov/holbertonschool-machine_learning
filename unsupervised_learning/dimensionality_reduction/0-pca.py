import numpy as np

def pca(X, var=0.95):
    """Perform PCA on X preserving var fraction of variance."""
    cov = np.cov(X, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    explained = eig_vals / np.sum(eig_vals)
    cumulative = np.cumsum(explained)

    nd = np.searchsorted(cumulative, var) + 1
    return eig_vecs[:, :nd]
