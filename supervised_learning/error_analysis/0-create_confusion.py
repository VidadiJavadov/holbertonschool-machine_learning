#!/usr/bin/env python3
"""creating conf matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """conf mat"""
    true_labels = np.argmax(labels, axis=1)
    true_logits = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    cm = np.zeros((classes, classes), dtype=int)
    for i, j in zip(true_labels, true_logits):
        cm[i][j] += 1

    return cm
