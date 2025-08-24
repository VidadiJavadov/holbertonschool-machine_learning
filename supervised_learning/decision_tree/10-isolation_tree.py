#!/usr/bin/env python3
"""Isolation Random Tree for outlier detection."""

import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree:
    """Isolation Random Tree classifier for outlier detection."""

    def __init__(self, max_depth=10, seed=0, root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        return str(self.root)

    def depth(self):
        return self.root.depth()

    def count_nodes(self, only_leaves=False):
        return self.root.count_nodes(only_leaves)

    def update_bounds(self):
        self.root.update_bounds()

    def get_leaves(self):
        return self.root.get_leaves_below()

    def update_predict(self):
        def node_depth_predict(x, node):
            """Recursive function to return leaf depth for each sample."""
            if isinstance(node, Leaf):
                return np.full(x.shape[0], node.depth)
            left_mask = x[:, node.feature] <= node.threshold
            right_mask = ~left_mask
            preds = np.empty(x.shape[0])
            if np.any(left_mask):
                preds[left_mask] = node_depth_predict(x[left_mask], node.left_child)
            if np.any(right_mask):
                preds[right_mask] = node_depth_predict(x[right_mask], node.right_child)
            return preds

        self.predict = lambda X: node_depth_predict(X, self.root)

    def np_extrema(self, arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Return a random feature and threshold for splitting."""
        features = np.arange(self.explanatory.shape[1])
        feature = self.rng.choice(features)
        vals = self.explanatory[node.subpopulation, feature]
        threshold = self.rng.uniform(np.min(vals), np.max(vals))
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        leaf_child = Leaf()
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        child = Node()
        child.depth = node.depth + 1
        child.subpopulation = sub_population
        return child

    def fit_node(self, node):
        node.feature, node.threshold = self.random_split_criterion(node)

        left_population = node.subpopulation & (self.explanatory[:, node.feature] <= node.threshold)
        right_population = node.subpopulation & (self.explanatory[:, node.feature] > node.threshold)

        is_left_leaf = node.depth + 1 >= self.max_depth or np.sum(left_population) <= self.min_pop
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = node.depth + 1 >= self.max_depth or np.sum(right_population) <= self.min_pop
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        self.explanatory = explanatory
        self.root.subpopulation = np.ones(explanatory.shape[0], dtype=bool)
        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(
                "  Training finished.\n"
                f"    - Depth                     : {self.depth()}\n"
                f"    - Number of nodes           : {self.count_nodes()}\n"
                f"    - Number of leaves          : {self.count_nodes(only_leaves=True)}"
            )
