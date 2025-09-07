#!/usr/bin/env python3
import numpy as np
"""one hot"""


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot encoded matrix."""
    return np.eye(classes if classes else np.max(labels) + 1)[labels]
