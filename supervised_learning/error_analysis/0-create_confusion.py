#!/usr/bin/env python3
"""creating conf matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """conf mat"""
    true_labels = np.argmax(labels)
    true_logits = np.argmax(logits)
    classes = true_labels.shape[1]
    cm = np.zeros((classes, classes), dtype=int)
    for i, j in zip(true_labels, true_logits):
        cm[i][j] += 1

    return cm
