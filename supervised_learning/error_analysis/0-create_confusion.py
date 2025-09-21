#!/usr/bin/env python3
"""creating conf matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """conf mat"""
    classes = len(set(labels[0]))
    cm = np.zeros((classes, classes), dtype=int)
    for i, j in zip(labels, logits):
        cm[i][j] += 1

    return cm
