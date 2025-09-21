#!/usr/bin/env python3
"""specificity"""
import numpy as np


def specificity(confusion):
    """specificity"""
    TN = np.sum(confusion)
    specificity = []
    for i in range(len(confusion[0])):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        spec = TN / (TN + FP)
        specificity.append(spec)
    return np.array(specificity)
