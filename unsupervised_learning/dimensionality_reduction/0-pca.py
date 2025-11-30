import numpy as np
"""PCA dimensionality reduction"""


def pca(X, var=0.95):
    """Performs PCA on dataset X while preserving given variance fraction."""
    
    # Compute covariance matrix
    cov = np.cov(X, rowvar=False)
    
    # Eigen decomposition
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    
    # Compute explained variance ratio
    explained = eig_vals / np.sum(eig_vals)
    cumulative = np.cumsum(explained)
    
    # Determine number of components needed
    nd = np.searchsorted(cumulative, var) + 1
    
    # Return the projection matrix W
    W = eig_vecs[:, :nd]
    return W
