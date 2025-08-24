#!/usr/bin/env python3
"""random forests"""


import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree

class Random_Forest():
    """random forests"""
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """init func"""
        self.numpy_predicts  = []
        self.target          = None
        self.numpy_preds     = None
        self.n_trees         = n_trees
        self.max_depth       = max_depth
        self.min_pop         = min_pop
        self.seed            = seed

    def predict(self, explanatory):
        """Predict class for each sample using the majority vote from all trees."""
        # Collect predictions from all trees
        all_preds = np.array([T(explanatory) for T in self.numpy_predicts])
        # Transpose so that rows correspond to samples
        all_preds = all_preds.T  # shape: (n_samples, n_trees)
        # Compute mode (most frequent value) along the tree axis
        from scipy.stats import mode
        mode_vals, _ = mode(all_preds, axis=1)
        return mode_vals.ravel()
