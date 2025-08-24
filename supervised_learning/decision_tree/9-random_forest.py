#!/usr/bin/env python3
"""Random Forest implementation using multiple Decision Trees."""

import numpy as np
from scipy.stats import mode
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest:
    """Random Forest classifier consisting of multiple Decision Trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initialize the Random Forest."""
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed
        self.numpy_predicts = []
        self.explanatory = None
        self.target = None

    def fit(self, explanatory, target, n_trees=None, verbose=0):
        """Train the Random Forest on the given dataset."""
        if n_trees is None:
            n_trees = self.n_trees

        self.explanatory = explanatory
        self.target = target
        self.numpy_predicts = []

        depths, nodes, leaves, accuracies = [], [], [], []

        for i in range(n_trees):
            tree = Decision_Tree(
                max_depth=self.max_depth,
                min_pop=self.min_pop,
                seed=self.seed + i
            )
            tree.fit(explanatory, target)
            self.numpy_predicts.append(tree.predict)
            depths.append(tree.depth())
            nodes.append(tree.count_nodes())
            leaves.append(tree.count_nodes(only_leaves=True))
            accuracies.append(tree.accuracy(tree.explanatory, tree.target))

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Mean depth                     : {np.mean(depths)}\n"
                f"    - Mean number of nodes           : {np.mean(nodes)}\n"
                f"    - Mean number of leaves          : {np.mean(leaves)}\n"
                f"    - Mean accuracy on training data : {np.mean(accuracies)}\n"
                f"    - Accuracy of the forest on td   : "
                f"{self.accuracy(self.explanatory, self.target)}"
            )

    def predict(self, explanatory):
        """Predict the class labels for the given data."""
        all_preds = np.array(
            [predict_func(explanatory) for predict_func in self.numpy_predicts]
        )
        all_preds = all_preds.T
        mode_vals, _ = mode(all_preds, axis=1)
        return mode_vals.ravel()

    def accuracy(self, test_explanatory, test_target):
        """Compute accuracy of the Random Forest on test data."""
        preds = self.predict(test_explanatory)
        return np.sum(preds == test_target) / test_target.size
